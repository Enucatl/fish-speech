# Cloning an Italian voice with Fish-Speech S2-Pro and LoRA finetuning

## Introduction

The goal of this project is to fine-tune S2-Pro to reproduce a specific
Italian speaker — a public figure whose voice is instantly recognisable to
Italian audiences: a distinctive tenor, a measured academic cadence,
a characteristic rhythm of clause and pause. The `s2-pro` model already knows
Italian, although as a _tier 3_ language. What it does not know is *this*
Italian, with this voice.

We set out to refine the model through LoRA finetuning, which is supported
(almost) out of the box.

## Why not zero-shot with a reference audio?

Fish-Speech supports zero-shot voice cloning: provide a few seconds of
reference audio and a transcript, and the model will attempt to reproduce
that speaker's voice for any new text. This works reasonably well, but
it will contain minor artifacts.

The model captures
some surface features of the voice but keeps generic 
prosody and intonation. The speaker's characteristic rhythm, the
way they elongate certain vowels, the specific quality of his consonants:
these can be represented in the weights.

## Improvements to the fine-tuning pipeline

Getting LoRA fine-tuning to work correctly on S2-Pro required several
fixes that are not in the upstream repository at the time of writing.

**`fix(deps): CUDA 12.9, updated wandb, protobuf override for RTX 5090`**
The RTX 5090 (Blackwell, sm_120) requires CUDA 12.9 and a matching
PyTorch build. The Dockerfile and `pyproject.toml` were updated
accordingly. Two additional dependency issues surfaced on this setup:
`wandb` needed a bump to `>=0.19.0`, and `descript-audiotools` pulls in
a `protobuf` version that conflicts with other dependencies — resolved
with a `uv` override pinning `protobuf>=3.20.0,<6.0.0`.

**`feat(lora): add `target_modules` with `fast_` prefix support`**
The original `setup_lora` function applied LoRA adapters to every linear
layer in the model unconditionally. We added a `target_modules` field to
`LoraConfig` that controls exactly which parts of the model receive
adapters. Valid values are `attention`, `mlp`, `embeddings`, `output` for
the slow transformer (and, for backwards compatibility, the fast
transformer too), and `fast_attention`, `fast_mlp`, `fast_embeddings`,
`fast_output` for the fast transformer only.

**`feat(callbacks): add `GradAccumProgressBar` and `AudioSampleCallback``**
`AudioSampleCallback` synthesises a sample audio file at each checkpoint during training, reusing the training
model in-place with no second copy loaded. Samples are saved alongside the checkpoints and logged to
TensorBoard, so you can listen to the voice evolve across training without
running a separate inference step.

## Dataset preparation

The training data consists of publicly available lecture recordings by the
target speaker. 55 recordings of approximately one hour each were manually
selected for quality and then processed through a four-stage pipeline.

### Prerequisites

Install the preprocessing dependencies:

```bash
uv sync --extra preprocess
```

You will also need:
- **ffmpeg** in your `PATH` (for audio conversion)
- A [Deepgram](https://deepgram.com/) API key (free tier is sufficient
  for a few hundred hours)
- Optionally, a [Google AI Studio](https://aistudio.google.com/) API key
  for Gemini transcript correction

### Stage 1 — Segmentation and transcription

Start with a folder of long audio files in any format ffmpeg can read
(MP3, MP4, M4A, FLAC, OGG, …). The preprocessing script at
`tools/preprocess/preprocess_audio.py` handles the full pipeline in a
single pass:

1. **Convert** — each file is converted to 24 kHz mono WAV via ffmpeg
   and cached in `_tmp_converted/` so the step is skipped on reruns.
2. **Diarize** *(optional, but strongly recommended)* — if a HuggingFace token is provided,
   pyannote 3.1 identifies the dominant speaker and discards all other
   segments (audience questions, applause, host intros). Nearby segments
   from the same speaker separated by less than 1.5 s are merged before
   the split step.
3. **Segment** — pydub silence detection splits each recording into
   chunks of 20–28 s. The algorithm looks for the latest silence gap
   within the preferred window; if none is found it falls back to the
   latest hard break (≥1.5 s silence) before 20 s; if still none it
   cuts hard at 28 s. Clips shorter than 3 s are discarded.
4. **Normalise** — each chunk is trimmed of leading/trailing silence,
   re-padded with 0.1 s on each side, and peak-normalised to −0.45 dBFS.
5. **Transcribe** — [Deepgram](https://deepgram.com/) nova-3 transcribes
   each chunk with 4 parallel workers and exponential-backoff retries.
   A `--keywords-file` (one term per line) can be passed to boost
   recognition of proper names or domain-specific vocabulary.
6. **Correct** *(optional, but strongly recommended)* — if a Google API key is provided, Gemini
   post-processes all transcripts for one audio file in a single
   context-aware call, correcting spelling and punctuation while
   preserving the keywords list. Batches of 30 chunks run in parallel.

With Gemini correction and a keywords file:

```bash
uv run python tools/preprocess/preprocess_audio.py \
    --input-dir /path/to/recordings \
    --output-dir data/raw/speaker \
    --deepgram-api-key $DEEPGRAM_API_KEY \
    --google-api-key $GOOGLE_API_KEY \
    --huggingface-token $HF_TOKEN \
    --keywords-file tools/preprocess/my_keywords.txt \
```

The script is resumable: converted and diarized WAVs are cached, and the
CSV is written only after all files complete. Output layout:

```
data/raw/speaker/
├── wavs/                   # normalised 20-28 s clips
│   ├── lecture01_0000.wav
│   ├── lecture01_0001.wav
│   └── ...
├── metadata.csv            # pipe-delimited, audio_file|text
├── debug.csv               # full audit trail with diarization detail
└── _tmp_converted/         # cached intermediates, safe to delete
```

### Stage 2 — Review and correct

Before committing the dataset it is worth spot-checking transcripts,
especially for proper names and domain-specific vocabulary that the ASR
may have mangled. The review server provides a browser interface for
this:

```bash
uv run python tools/preprocess/review_server.py \
    --output-dir data/raw/speaker
```

Open `http://localhost:8765` to play each clip alongside its raw and
Gemini-corrected transcript. Edits are saved to `corrections.csv`.
Entire source files can be masked from the dataset via `masked_sources.txt`, so
that it's easy to exclude low-quality samples.

### Stage 3 — Format conversion for Fish-Speech

Fish-Speech expects pairs of `.wav` and `.lab` (plain-text transcript)
files in a flat directory. The conversion script symlinks the audio
(avoiding duplication of large files) and writes each transcript as a
`.lab`:

```bash
uv run python tools/preprocess/prepare_dataset.py \
    --metadata data/raw/speaker/metadata.csv \
    --output-dir data/speaker
```

The script reads both `metadata.csv` and `corrections.csv` (if present),
with corrections taking precedence, and writes to `data/speaker/`.

### Stage 4 — VQ token extraction

The DAC codec tokenises each clip into discrete codes that the model
trains on. This is a one-time pass and takes roughly 1–2× realtime on
GPU:

```bash
uv run python tools/vqgan/extract_vq.py data/speaker \
    --num-workers 1 --batch-size 16 \
    --config-name modded_dac_vq \
    --checkpoint-path checkpoints/s2-pro/codec.pth
```

### Stage 5 — Protobuf sharding

The data loader consumes protobuf shards. All clips are packed into a
single group under `data/protos/`:

```bash
uv run python tools/llama/build_dataset.py \
    --input data/speaker --output data/protos \
    --text-extension .lab --num-workers 4
```

The resulting dataset contains **8,011 clips** with an average duration
of **24.8 seconds**, for a total of approximately **55 hours** of speech
from a single speaker. All clips land in a single proto group, which has
implications for how the data loader samples them — more on that in the
training section.
