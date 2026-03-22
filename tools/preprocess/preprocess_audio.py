"""Preprocess audio for F5-TTS finetuning.

Pipeline:
  1. Convert input audio to 24 kHz mono WAV (via ffmpeg/pydub).
  2. (Optional) Speaker diarization via pyannote to keep only the main speaker,
     discarding intros, outros, applause, and other-speaker segments.
  3. Split into chunks of 20-28 s using VAD-aware silence detection.
  4. Trim silence edges and peak-normalise each chunk.
  5. Transcribe each chunk with Deepgram (nova-3 by default).
  6. Optionally post-process transcripts with Gemini for spelling / punctuation.
  7. Write a pipe-delimited CSV ready for prepare_csv_wavs.py.

Usage examples
--------------
# Transcribe only
python preprocess_audio.py \\
    --input-dir /raw/audio \\
    --output-dir data/italian_tts \\
    --deepgram-api-key $DEEPGRAM_API_KEY

# With Gemini post-processing
python preprocess_audio.py \\
    --input-dir /raw/audio \\
    --output-dir data/italian_tts \\
    --deepgram-api-key $DEEPGRAM_API_KEY \\
    --google-api-key $GOOGLE_API_KEY
"""

from __future__ import annotations

import csv
import logging
import math
import subprocess
import sys
from pathlib import Path

import click
from deepgram import DeepgramClient
from pydub import AudioSegment
from pydub.silence import detect_nonsilent, detect_silence

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

TARGET_SR = 24_000  # Hz required by F5-TTS

# Chunking thresholds
CHUNK_MIN_MS = 20_000       # preferred lower bound (20 s)
CHUNK_MAX_MS = 28_000       # hard upper bound (28 s)
CHUNK_ABS_MIN_MS = 3_000    # minimum keepable clip (3 s)

# A silence longer than this is treated as a "hard break" (topic change,
# paragraph boundary) and may trigger an early cut even below CHUNK_MIN_MS.
HARD_BREAK_MS = 1_500       # 1.5 s

# Diarization: gaps between same-speaker segments shorter than this are merged
# so the splitter sees continuous runs rather than many small islands.
DIARIZATION_MERGE_GAP_MS = 1_500   # 1.5 s

# Silence trimming: strip leading/trailing silence then re-pad to this amount.
SILENCE_PAD_MS = 100        # 0.1 s of silence kept at each edge

# Peak normalisation target (linear, where 1.0 = full scale).
NORM_PEAK = 0.95
NORM_PEAK_DBFS = 20 * math.log10(NORM_PEAK)  # ≈ -0.446 dBFS


def _to_wav(src: Path, dst: Path) -> None:
    """Convert any audio file to 24 kHz mono WAV using ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-ar", str(TARGET_SR),
        "-ac", "1",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {src}:\n{result.stderr.decode()}"
        )


def _load_diarization_pipeline(huggingface_token: str):
    """Load the pyannote diarization pipeline once (call at startup, not per file)."""
    import torch
    from pyannote.audio import Pipeline as _PyannotePipeline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Loading diarization pipeline on %s …", device)
    pipeline = _PyannotePipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=huggingface_token,
    )
    pipeline.to(device)
    return pipeline


def _extract_main_speaker(
    wav_path: Path, diarization_pipeline,
) -> tuple[AudioSegment, list[dict]]:
    """Run pyannote speaker diarization and return audio of the dominant speaker only.

    Returns (audio, audit_rows) where audit_rows is a list of dicts describing
    every raw diarization segment with keys:
        source_file, type="diarization_segment", speaker, start_s, end_s,
        duration_s, kept (True/False), chunk_file, raw_transcript, corrected_transcript

    The dominant speaker is whichever speaker has the highest total speaking
    time — in a conference recording that will be the presenter.  Applause,
    crowd noise, and the host's intro/outro are all filtered out.

    Nearby segments from the same speaker are merged when the gap between them
    is ≤ DIARIZATION_MERGE_GAP_MS so that the splitter sees natural long runs
    of speech rather than many tiny islands.
    """
    log.info("   Running speaker diarization …")
    diarization = diarization_pipeline(str(wav_path)).speaker_diarization

    # Accumulate segments per speaker
    speaker_segments: dict[str, list[tuple[float, float]]] = {}
    all_raw_segments: list[tuple[float, float, str]] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_segments.setdefault(speaker, []).append((turn.start, turn.end))
        all_raw_segments.append((turn.start, turn.end, speaker))

    if not speaker_segments:
        log.warning("   Diarization returned no segments — using full audio.")
        return AudioSegment.from_wav(str(wav_path)), []

    # Pick the speaker with most total speaking time
    def _total(segs: list[tuple[float, float]]) -> float:
        return sum(e - s for s, e in segs)

    main_speaker = max(speaker_segments, key=lambda sp: _total(speaker_segments[sp]))
    segments = sorted(speaker_segments[main_speaker])
    total_dur = _total(segments)
    log.info(
        "   Main speaker: %s  (%.1f s across %d segment(s))",
        main_speaker, total_dur, len(segments),
    )

    # Merge segments whose gap is small enough that cutting would be awkward
    merged: list[tuple[float, float]] = []
    for start, end in segments:
        if merged and (start - merged[-1][1]) * 1000 <= DIARIZATION_MERGE_GAP_MS:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    # Concatenate the kept regions into a single AudioSegment
    audio = AudioSegment.from_wav(str(wav_path))
    kept = AudioSegment.empty()
    for start, end in merged:
        kept += audio[int(start * 1000): int(end * 1000)]

    log.info(
        "   Kept %.1f s of %.1f s total (%.0f%% discarded).",
        len(kept) / 1000,
        len(audio) / 1000,
        100 * (1 - len(kept) / len(audio)),
    )

    # Build audit rows for every raw segment
    kept_set = set(id(s) for s in segments)  # not reliable across lists; use interval check
    def _is_kept(start: float, end: float) -> bool:
        return any(s <= start and end <= e for s, e in segments)

    audit_rows = [
        {
            "source_file": wav_path.name,
            "type": "diarization_segment",
            "speaker": spk,
            "start_s": f"{start:.3f}",
            "end_s": f"{end:.3f}",
            "duration_s": f"{end - start:.3f}",
            "kept": str(_is_kept(start, end)),
            "chunk_file": "",
            "raw_transcript": "",
            "corrected_transcript": "",
        }
        for start, end, spk in sorted(all_raw_segments)
    ]

    return kept, audit_rows


def _trim_chunk(chunk: AudioSegment) -> AudioSegment:
    """Strip leading/trailing silence, then re-pad with SILENCE_PAD_MS on each side.

    Uses the same dBFS threshold as the splitter so behaviour is consistent.
    Falls back to the original segment if it turns out to be fully silent.
    """
    regions = detect_nonsilent(chunk, min_silence_len=100, silence_thresh=chunk.dBFS - 16)
    if not regions:
        return chunk  # fully silent — caller will discard if too short
    speech_start = regions[0][0]
    speech_end = regions[-1][1]
    trimmed = chunk[speech_start:speech_end]
    silence_pad = AudioSegment.silent(duration=SILENCE_PAD_MS, frame_rate=TARGET_SR)
    padded = silence_pad + trimmed + silence_pad

    # Peak-normalise to NORM_PEAK (skip if silent to avoid -inf dBFS)
    if padded.max_dBFS > -float("inf"):
        padded = padded.apply_gain(NORM_PEAK_DBFS - padded.max_dBFS)
    return padded


def _split_audio(wav_path: Path, out_dir: Path, stem: str) -> list[Path]:
    """Split a WAV file into chunks targeting 20-28 s with a natural tail.

    Cut-point priority for each chunk starting at `start_ms`:
      1. Latest normal silence gap in [start + CHUNK_MIN_MS, start + CHUNK_MAX_MS]
         → produces the bulk of clips in the 20-28 s range.
      2. Latest *hard break* (silence ≥ HARD_BREAK_MS) in
         [start + CHUNK_ABS_MIN_MS, start + CHUNK_MIN_MS)
         → produces the occasional short clip visible in a good distribution.
      3. Hard cut at start + CHUNK_MAX_MS if no silence found at all.

    The final leftover (end of file) is kept as-is if ≥ CHUNK_ABS_MIN_MS,
    which also contributes to the short-clip tail of the distribution.
    """
    audio = AudioSegment.from_wav(str(wav_path))
    total_ms = len(audio)

    # detect_silence returns [(sil_start_ms, sil_end_ms), ...]
    silence_regions = detect_silence(
        audio,
        min_silence_len=400,
        silence_thresh=audio.dBFS - 16,
    )

    if not silence_regions:
        # No silence at all — one long chunk or discard if too short
        if total_ms >= CHUNK_ABS_MIN_MS:
            path = out_dir / f"{stem}_0000.wav"
            audio.export(str(path), format="wav")
            return [path]
        log.warning("%s is too short and has no silence, skipping", wav_path.name)
        return []

    # Pre-compute cut-point candidates: (midpoint_ms, silence_duration_ms)
    cut_points: list[tuple[int, int]] = [
        ((s + e) // 2, e - s) for s, e in silence_regions
    ]

    chunks: list[Path] = []
    start_ms = 0
    chunk_idx = 0

    while start_ms < total_ms:
        remaining_ms = total_ms - start_ms

        if remaining_ms <= CHUNK_MAX_MS:
            # Everything that's left becomes the final chunk (may be short)
            end_ms = total_ms
        else:
            target_end = start_ms + CHUNK_MAX_MS

            # Priority 1: latest normal gap in preferred window
            preferred = [
                mid for mid, _ in cut_points
                if start_ms + CHUNK_MIN_MS <= mid <= target_end
            ]

            # Priority 2: latest hard break before preferred window
            early = [
                mid for mid, dur in cut_points
                if dur >= HARD_BREAK_MS
                and start_ms + CHUNK_ABS_MIN_MS <= mid < start_ms + CHUNK_MIN_MS
            ]

            if preferred:
                end_ms = max(preferred)
            elif early:
                end_ms = max(early)
            else:
                end_ms = target_end  # hard cut

        duration_ms = end_ms - start_ms
        if duration_ms < CHUNK_ABS_MIN_MS:
            log.debug("Discarding %.1f s tail of %s", duration_ms / 1000, wav_path.name)
            break

        chunk = _trim_chunk(audio[start_ms:end_ms])
        if len(chunk) < CHUNK_ABS_MIN_MS:
            log.debug("Chunk shrank below minimum after trimming, discarding.")
            start_ms = end_ms
            continue
        chunk_path = out_dir / f"{stem}_{chunk_idx:04d}.wav"
        chunk.export(str(chunk_path), format="wav")
        chunks.append(chunk_path)
        chunk_idx += 1
        start_ms = end_ms

    return chunks


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def _load_keywords(path: Path) -> list[str]:
    """Read a keywords file (one keyword/phrase per line, # comments ignored)."""
    keywords = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            keywords.append(line)
    return keywords


def _transcribe(
    wav_path: Path,
    deepgram: DeepgramClient,
    model: str,
    language: str,
    smart_format: bool,
    keywords: list[str],
    max_retries: int = 5,
) -> str:
    import time
    delay = 2.0
    for attempt in range(max_retries):
        try:
            with wav_path.open("rb") as fh:
                response = deepgram.listen.v1.media.transcribe_file(
                    request=fh.read(),
                    model=model,
                    language=language,
                    smart_format=smart_format,
                    keyterm=keywords or None,
                    request_options={"timeout_in_seconds": 300},
                )
            return response.results.channels[0].alternatives[0].transcript
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            log.warning("   Transcription attempt %d/%d failed for %s (%s), retrying in %.0fs …",
                        attempt + 1, max_retries, wav_path.name, exc, delay)
            time.sleep(delay)
            delay *= 2


def _gemini_correct_batch(
    transcripts: list[str], language: str, model_name: str, keywords: list[str], google_api_key: str
) -> list[str]:
    """Send all chunks for one audio file to Gemini in a single call for speed and cross-chunk context."""
    import google.genai as genai
    import google.genai.types as genai_types

    keywords_section = ""
    if keywords:
        keywords_section = f"""
The following is a reference list of proper names, technical terms, and unusual
words known to appear in this speaker's content:
{", ".join(keywords)}

When reviewing the transcripts, actively consider whether any word or phrase that
looks misspelled, garbled, or out of place might actually be one of these terms
that the speech recogniser failed to catch correctly. Substitute the correct form
from the list wherever the context supports it. Also preserve the exact spelling
of any of these terms that were already transcribed correctly.
"""

    numbered = "\n\n".join(f"[{i}]\n{t}" for i, t in enumerate(transcripts))
    correction_prompt = f"""Below are {len(transcripts)} consecutive chunks from the same audio recording,
each labelled with a zero-based index in square brackets.

The two-letter code for the language is: {language}.
{keywords_section}
For each chunk, correct errors in spelling and punctuation and fix any mistranscribed words,
staying as close as possible to the original phrasing.

You MUST return exactly {len(transcripts)} chunks.
Output format — reproduce every index label exactly as it appears in the input, followed by the corrected text.
Do not merge or split chunks. Do not add any commentary before, between, or after the chunks.

Example input:
[0]
Ciao a tutti benvenuti al podcast

[1]
oggi parliamo di storia medievale

Example output:
[0]
Ciao a tutti, benvenuti al podcast.

[1]
Oggi parliamo di storia medievale.

Now correct the following:

{numbered}
"""
    client = genai.Client(api_key=google_api_key)
    response = client.models.generate_content(
        model=model_name,
        contents=correction_prompt,
        config=genai_types.GenerateContentConfig(
            temperature=0.7,
            safety_settings=[
                genai_types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                genai_types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                genai_types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                genai_types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
            ],
        ),
    )

    # Parse response: split on index markers [0], [1], ...
    import re
    parts = re.split(r"\[(\d+)\]\s*", response.text.strip())
    # parts = ['', '0', 'text0', '1', 'text1', ...]
    corrected: dict[int, str] = {}
    for idx in range(1, len(parts) - 1, 2):
        chunk_idx = int(parts[idx])
        corrected[chunk_idx] = parts[idx + 1].strip()

    # Fall back to original if a chunk is missing from the response
    return [corrected.get(i, transcripts[i]) for i in range(len(transcripts))]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Directory containing raw audio files (any format ffmpeg can read). "
         "Mutually exclusive with --input-file.",
)
@click.option(
    "--input-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Single audio file to process. Mutually exclusive with --input-dir.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Destination directory.  WAV chunks go in <output-dir>/wavs/, "
         "CSV written to <output-dir>/metadata.csv. "
         "Defaults to a sibling directory of the input named 'tts_dataset'.",
)
@click.option("--deepgram-api-key", required=True, envvar="DEEPGRAM_API_KEY")
@click.option("--deepgram-model", default="nova-3", show_default=True)
@click.option(
    "--google-api-key",
    default=None,
    envvar="GOOGLE_API_KEY",
    help="If provided, use Gemini to post-process transcripts.",
)
@click.option("--google-model", default="gemini-3-flash-preview", show_default=True)
@click.option("--language", default="it", show_default=True)
@click.option("--smart-format/--no-smart-format", default=True, show_default=True)
@click.option(
    "--huggingface-token",
    default=None,
    envvar="HUGGINGFACE_TOKEN",
    help="HuggingFace token for pyannote diarization models. "
         "If omitted, diarization is skipped and the full audio is used.",
)
@click.option(
    "--keywords-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Plain-text file with one keyword/phrase per line. "
         "Passed to Deepgram for boosted recognition and to Gemini to preserve spelling.",
)
@click.option(
    "--glob",
    "file_glob",
    default="**/*",
    show_default=True,
    help="Glob pattern relative to --input-dir to select files (ignored for --input-file).",
)
@click.option(
    "--exclude-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Plain-text file listing filenames (one per line) to skip. "
         "Matched against the basename of each input file.",
)
def main(
    input_dir: Path | None,
    input_file: Path | None,
    output_dir: Path | None,
    deepgram_api_key: str,
    deepgram_model: str,
    google_api_key: str | None,
    google_model: str,
    language: str,
    smart_format: bool,
    huggingface_token: str | None,
    keywords_file: Path | None,
    file_glob: str,
    exclude_file: Path | None,
) -> None:
    """Convert, split, and transcribe audio for F5-TTS finetuning."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        stream=sys.stderr,
    )

    if input_dir is None and input_file is None:
        raise click.UsageError("Provide either --input-dir or --input-file.")
    if input_dir is not None and input_file is not None:
        raise click.UsageError("--input-dir and --input-file are mutually exclusive.")

    # Resolve output directory
    if output_dir is None:
        base = input_file.parent if input_file else input_dir.parent  # type: ignore[union-attr]
        output_dir = base / "tts_dataset"

    wavs_dir = output_dir / "wavs"
    tmp_dir = output_dir / "_tmp_converted"
    wavs_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metadata.csv"

    # Collect input files
    audio_extensions = {
        ".mp3", ".mp4", ".m4a", ".aac", ".ogg", ".flac",
        ".wav", ".wma", ".opus", ".webm",
    }
    if input_file is not None:
        input_files = [input_file]
    else:
        input_files = sorted(
            p for p in input_dir.glob(file_glob)  # type: ignore[union-attr]
            if p.is_file() and p.suffix.lower() in audio_extensions
        )
    if not input_files:
        log.error("No audio files found.")
        sys.exit(1)

    # Apply exclusion list
    n_excluded = 0
    if exclude_file is not None:
        excluded = {
            line.strip() for line in exclude_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        }
        before = len(input_files)
        input_files = [f for f in input_files if f.name not in excluded]
        n_excluded = before - len(input_files)

    if n_excluded:
        log.info("Will process %d file(s)  (%d excluded, %d total found)",
                 len(input_files), n_excluded, len(input_files) + n_excluded)
    else:
        log.info("Will process %d file(s)", len(input_files))

    # Configure Gemini once if requested
    if google_api_key is not None:
        import google.genai as genai
        _genai_client = genai.Client(api_key=google_api_key)
        available = {m.name.removeprefix("models/") for m in _genai_client.models.list()}
        if google_model not in available:
            log.error(
                "Model '%s' not available. Available models:\n%s",
                google_model,
                "\n".join(sorted(available)),
            )
            sys.exit(1)
        log.info("Gemini post-processing enabled with model '%s'", google_model)

    keywords: list[str] = _load_keywords(keywords_file) if keywords_file else []
    if keywords:
        log.info("Loaded %d keyword(s) from %s", len(keywords), keywords_file)

    deepgram = DeepgramClient(api_key=deepgram_api_key)

    # Load diarization pipeline once for all files
    diarization_pipeline = None
    if huggingface_token is not None:
        diarization_pipeline = _load_diarization_pipeline(huggingface_token)

    rows: list[dict[str, str]] = []
    debug_rows: list[dict[str, str]] = []

    for audio_file in input_files:
        stem = audio_file.stem
        log.info("── Processing: %s", audio_file.name)

        # 1. Convert to 24 kHz mono WAV
        converted_wav = tmp_dir / f"{stem}.wav"
        if not converted_wav.exists():
            log.info("   Converting to 24 kHz mono WAV …")
            try:
                _to_wav(audio_file, converted_wav)
            except RuntimeError as exc:
                log.error("   Conversion failed: %s", exc)
                continue
        else:
            log.info("   Reusing existing converted WAV.")

        # 2. (Optional) Speaker diarization — keep main speaker only
        diarization_audit: list[dict] = []
        diarized_wav = tmp_dir / f"{stem}_diarized.wav"
        if diarization_pipeline is not None:
            if not diarized_wav.exists():
                try:
                    main_speaker_audio, diarization_audit = _extract_main_speaker(
                        converted_wav, diarization_pipeline
                    )
                    main_speaker_audio.export(str(diarized_wav), format="wav")
                except Exception as exc:
                    log.error("   Diarization failed: %s", exc)
                    sys.exit(1)
            else:
                log.info("   Reusing existing diarized WAV.")
            source_wav = diarized_wav
        else:
            source_wav = converted_wav


        debug_rows.extend(diarization_audit)

        # 3. Split into chunks
        log.info("   Splitting into chunks …")
        chunks = _split_audio(source_wav, wavs_dir, stem)
        if not chunks:
            log.warning("   No chunks produced, skipping.")
            continue
        log.info("   Produced %d chunk(s).", len(chunks))

        # 4. Transcribe all chunks in parallel
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _transcribe_chunk(chunk_path: Path) -> tuple[Path, str | None]:
            try:
                t = _transcribe(chunk_path, deepgram, deepgram_model, language, smart_format, keywords)
                return chunk_path, t if t.strip() else None
            except Exception as exc:  # noqa: BLE001
                log.error("   Transcription failed for %s: %s", chunk_path.name, exc)
                return chunk_path, None

        log.info("   Transcribing %d chunk(s) in parallel …", len(chunks))
        chunk_transcripts: list[tuple[Path, str]] = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_transcribe_chunk, p): p for p in chunks}
            # Preserve original order
            results: dict[Path, str | None] = {}
            for fut in as_completed(futures):
                path, text = fut.result()
                results[path] = text
        for chunk_path in chunks:
            text = results.get(chunk_path)
            if text is None:
                continue
            chunk_transcripts.append((chunk_path, text))

        if not chunk_transcripts:
            continue

        raw_transcripts = [t for _, t in chunk_transcripts]

        # 5. Optionally batch-correct with Gemini (parallel batches of 30)
        GEMINI_BATCH = 30
        corrected_transcripts = list(raw_transcripts)
        if google_api_key is not None:
            batches = [
                (i, raw_transcripts[i: i + GEMINI_BATCH])
                for i in range(0, len(raw_transcripts), GEMINI_BATCH)
            ]
            n_batches = len(batches)
            log.info("   Applying Gemini correction: %d chunk(s) in %d parallel batch(es) …",
                     len(raw_transcripts), n_batches)
            try:
                corrected_map: dict[int, list[str]] = {}
                def _correct_batch(args):
                    start, batch = args
                    return start, _gemini_correct_batch(batch, language, google_model, keywords, google_api_key)
                with ThreadPoolExecutor(max_workers=n_batches) as pool:
                    for start, result in pool.map(_correct_batch, batches):
                        corrected_map[start] = result
                corrected_transcripts = []
                for i in range(0, len(raw_transcripts), GEMINI_BATCH):
                    corrected_transcripts.extend(corrected_map[i])
            except Exception as exc:  # noqa: BLE001
                log.warning("   Gemini correction failed (%s), keeping raw transcripts.", exc)

        for i, (chunk_path, _) in enumerate(chunk_transcripts):
            corrected = corrected_transcripts[i]
            rows.append({"audio_file": str(chunk_path.resolve()), "text": corrected})
            debug_rows.append({
                "source_file": audio_file.name,
                "type": "transcript_chunk",
                "speaker": "",
                "start_s": "",
                "end_s": "",
                "duration_s": "",
                "kept": "True",
                "chunk_file": str(chunk_path.resolve()),
                "raw_transcript": raw_transcripts[i],
                "corrected_transcript": corrected,
            })

    if not rows:
        log.error("No transcriptions produced. CSV not written.")
        sys.exit(1)

    # Write metadata.csv
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["audio_file", "text"], delimiter="|")
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote %d row(s) to %s", len(rows), csv_path)

    # Write debug.csv
    debug_fields = [
        "source_file", "type", "speaker", "start_s", "end_s", "duration_s",
        "kept", "chunk_file", "raw_transcript", "corrected_transcript",
    ]
    debug_path = output_dir / "debug.csv"
    with debug_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=debug_fields, delimiter="|")
        writer.writeheader()
        writer.writerows(debug_rows)
    log.info("Wrote debug log (%d row(s)) to %s", len(debug_rows), debug_path)

    click.echo(str(csv_path))


if __name__ == "__main__":
    main()
