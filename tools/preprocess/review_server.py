"""
Simple review server for the TTS preprocessing pipeline output.

Reads debug.csv from an output directory, serves audio chunks with their raw
and corrected transcriptions, collects manual corrections into corrections.csv,
and allows masking entire source files via masked_sources.txt.

Usage:
    uv run src/f5_tts/train/datasets/review_server.py --output-dir data/processed/barbero-test
"""
import csv
from collections import Counter
from datetime import datetime
from pathlib import Path

import click
import soundfile as sf
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse

OUTPUT_DIR: Path = None  # set at startup


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_debug(output_dir: Path) -> list[dict]:
    path = output_dir / "debug.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="|"))


def _load_corrections(output_dir: Path) -> dict[str, str]:
    """Return {chunk_file: corrected_text}."""
    path = output_dir / "corrections.csv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return {row["chunk_file"]: row["corrected_transcript"] for row in csv.DictReader(fh, delimiter="|")}


def _save_correction(output_dir: Path, chunk_file: str, text: str) -> None:
    path = output_dir / "corrections.csv"
    corrections = _load_corrections(output_dir)
    corrections[chunk_file] = text
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["chunk_file", "corrected_transcript", "saved_at"], delimiter="|")
        writer.writeheader()
        for cf, ct in corrections.items():
            writer.writerow({"chunk_file": cf, "corrected_transcript": ct, "saved_at": datetime.now().isoformat()})


def _load_masked(output_dir: Path) -> set[str]:
    """Return set of masked source filenames."""
    path = output_dir / "masked_sources.txt"
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def _duration_histogram(chunk_rows: list[dict], bar_width: int = 40) -> str:
    """Return an HTML <pre> block with an ASCII duration histogram."""
    durations = []
    for row in chunk_rows:
        path = row["chunk_file"]
        try:
            durations.append(sf.info(path).duration)
        except Exception:
            pass
    if not durations:
        return ""
    buckets = Counter(int(d) for d in durations)
    lo, hi = min(buckets), max(buckets)
    max_count = max(buckets.values())
    total_h = sum(durations) / 3600
    mean_d = sum(durations) / len(durations)
    lines = [
        f"Duration distribution  {len(durations)} chunks · "
        f"min {min(durations):.0f}s · max {max(durations):.0f}s · "
        f"mean {mean_d:.1f}s · total {total_h:.1f}h",
        "",
    ]
    for s in range(lo, hi + 1):
        count = buckets.get(s, 0)
        bar = "█" * int(count / max_count * bar_width)
        lines.append(f"{s:3d}s │{bar:<{bar_width}s} {count}")
    return "<pre style='font-size:0.8rem;line-height:1.4;background:#1a1a2e;color:#adf;padding:1rem 1.5rem;border-radius:8px;overflow-x:auto;'>" + "\n".join(lines) + "</pre>"


def _toggle_mask(output_dir: Path, source_file: str) -> bool:
    """Toggle mask for source_file. Returns True if now masked."""
    masked = _load_masked(output_dir)
    if source_file in masked:
        masked.discard(source_file)
        now_masked = False
    else:
        masked.add(source_file)
        now_masked = True
    path = output_dir / "masked_sources.txt"
    path.write_text("\n".join(sorted(masked)) + ("\n" if masked else ""))
    return now_masked


# ---------------------------------------------------------------------------
# HTML — index page
# ---------------------------------------------------------------------------

INDEX_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>TTS Review — {output_dir}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: system-ui, sans-serif; margin: 0; background: #f5f5f5; color: #222; }}
  header {{ background: #1a1a2e; color: #eee; padding: 1rem 2rem; position: sticky; top: 0; z-index: 10; }}
  header h1 {{ margin: 0 0 0.2rem; font-size: 1.1rem; }}
  header .stats {{ font-size: 0.85rem; opacity: 0.7; }}
  main {{ max-width: 900px; margin: 0 auto; padding: 1.5rem; }}
  table {{ width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
  th {{ background: #f0f0f0; padding: 0.6rem 1rem; text-align: left; font-size: 0.82rem; color: #555; font-weight: 600; }}
  td {{ padding: 0.55rem 1rem; border-top: 1px solid #f0f0f0; font-size: 0.85rem; }}
  tr.masked td {{ color: #aaa; background: #fdf2f2; }}
  tr:hover td {{ background: #f0f7ff; }}
  a {{ color: #2980b9; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .badge {{ font-size: 0.7rem; padding: 2px 8px; border-radius: 99px; font-weight: 600; }}
  .badge.masked {{ background: #fadbd8; color: #922b21; }}
  .badge.ok {{ background: #d5f5e3; color: #1e8449; }}
  .badge.corrected {{ background: #d6eaf8; color: #1a5276; }}
</style>
</head>
<body>
<header>
  <h1>TTS Review</h1>
  <div class="stats">{output_dir} &mdash; {n_sources} sources &mdash; {n_masked} masked &mdash; {n_chunks} chunks &mdash; {n_corrected} manually corrected</div>
</header>
<main>
{histogram}
<table>
  <tr>
    <th>#</th>
    <th>Source file</th>
    <th>Chunks</th>
    <th>Corrections</th>
    <th>Status</th>
  </tr>
  {rows}
</table>
</main>
</body>
</html>"""

INDEX_ROW = """<tr class="{row_class}">
  <td>{i}</td>
  <td><a href="/source/{source_file}">{source_file}</a></td>
  <td>{n_chunks}</td>
  <td>{n_corrected}</td>
  <td>{status_badge}</td>
</tr>"""


# ---------------------------------------------------------------------------
# HTML — source detail page
# ---------------------------------------------------------------------------

DETAIL_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Review — {source_file}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: system-ui, sans-serif; margin: 0; background: #f5f5f5; color: #222; }}
  header {{ background: #1a1a2e; color: #eee; padding: 1rem 2rem; position: sticky; top: 0; z-index: 10; display: flex; align-items: center; gap: 2rem; flex-wrap: wrap; }}
  header h1 {{ margin: 0; font-size: 1rem; font-weight: 600; flex: 1; }}
  header .stats {{ font-size: 0.82rem; opacity: 0.7; }}
  .nav {{ display: flex; gap: 0.6rem; align-items: center; }}
  .nav a {{ color: #aad4f5; font-size: 0.85rem; text-decoration: none; }}
  .nav a:hover {{ color: #fff; }}
  main {{ max-width: 1100px; margin: 0 auto; padding: 1.5rem; }}

  .source-header {{ display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem; padding: 0.8rem 1.2rem; background: #fff; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
  .source-header.masked {{ background: #fdf2f2; border-left: 4px solid #e74c3c; }}
  .source-header.active {{ border-left: 4px solid #2ecc71; }}
  .btn-mask {{ padding: 0.35rem 0.9rem; border: none; border-radius: 5px; cursor: pointer; font-size: 0.8rem; font-weight: 700; }}
  .btn-mask.unmask {{ background: #e74c3c; color: #fff; }}
  .btn-mask.unmask:hover {{ background: #c0392b; }}
  .btn-mask.mask {{ background: #ecf0f1; color: #555; }}
  .btn-mask.mask:hover {{ background: #dfe6e9; }}

  .chunk {{ background: #fff; border-radius: 8px; margin-bottom: 1rem; padding: 1.2rem 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
  .chunk.corrected {{ border-left: 4px solid #2ecc71; }}
  .chunk.pending   {{ border-left: 4px solid #e67e22; }}
  .source-group.masked .chunk {{ opacity: 0.45; pointer-events: none; }}
  .chunk-header {{ display: flex; align-items: center; gap: 1rem; margin-bottom: 0.8rem; flex-wrap: wrap; }}
  .chunk-id {{ font-size: 0.8rem; font-weight: 600; color: #888; text-transform: uppercase; letter-spacing: .04em; }}
  .badge {{ font-size: 0.7rem; padding: 2px 8px; border-radius: 99px; font-weight: 600; }}
  .badge.saved   {{ background: #d5f5e3; color: #1e8449; }}
  .badge.pending {{ background: #fdebd0; color: #a04000; }}
  audio {{ width: 100%; margin-bottom: 0.8rem; }}
  .transcripts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 0.8rem; }}
  @media (max-width: 700px) {{ .transcripts {{ grid-template-columns: 1fr; }} }}
  .transcript-box label {{ display: block; font-size: 0.72rem; font-weight: 600; color: #888; text-transform: uppercase; letter-spacing: .05em; margin-bottom: 0.3rem; }}
  .transcript-box .raw-text {{ font-size: 0.88rem; color: #555; background: #f9f9f9; border: 1px solid #e0e0e0; border-radius: 5px; padding: 0.6rem 0.8rem; min-height: 4rem; line-height: 1.5; }}
  textarea {{ width: 100%; font-size: 0.88rem; line-height: 1.5; border: 1px solid #ccc; border-radius: 5px; padding: 0.6rem 0.8rem; resize: vertical; min-height: 4rem; font-family: inherit; }}
  textarea:focus {{ outline: none; border-color: #3498db; box-shadow: 0 0 0 2px rgba(52,152,219,.2); }}
  .actions {{ display: flex; gap: 0.6rem; align-items: center; }}
  button {{ padding: 0.4rem 1rem; border: none; border-radius: 5px; cursor: pointer; font-size: 0.85rem; font-weight: 600; }}
  .btn-save  {{ background: #3498db; color: #fff; }}
  .btn-save:hover  {{ background: #2980b9; }}
  .btn-reset {{ background: #ecf0f1; color: #555; }}
  .btn-reset:hover {{ background: #dfe6e9; }}
  .save-msg {{ font-size: 0.78rem; color: #27ae60; display: none; }}

  .diar-section {{ margin-bottom: 2rem; }}
  .diar-section h2 {{ font-size: 0.9rem; color: #888; text-transform: uppercase; letter-spacing: .05em; margin-bottom: 0.5rem; }}
  table.diar {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
  table.diar th {{ background: #f0f0f0; padding: 0.5rem 0.8rem; text-align: left; font-weight: 600; color: #555; }}
  table.diar td {{ padding: 0.4rem 0.8rem; border-top: 1px solid #f0f0f0; }}
  table.diar tr.discarded td {{ color: #aaa; }}
  table.diar tr.main-speaker td {{ color: #1e8449; font-weight: 500; }}
</style>
</head>
<body>
<header>
  <h1>{source_file}</h1>
  <span class="stats">{n_chunks} chunks &mdash; {n_corrected} corrected</span>
  <div class="nav">
    {prev_link}
    <a href="/">&#8617; index</a>
    {next_link}
  </div>
</header>
<main>
{diarization_section}
<div class="{grp_class}" id="grp_{source_file}">
  <div class="source-header {hdr_class}" id="hdr_{source_file}">
    <span style="flex:1; font-weight:600; font-size:0.9rem;">{source_file}</span>
    <button class="btn-mask {btn_class}" id="btn_{source_file}" onclick="toggleMask('{source_file}')">{btn_text}</button>
  </div>
  {chunks_html}
</div>
</main>
<script>
function saveCorrection(chunkFile, geminiText) {{
  const ta  = document.getElementById('ta_'  + CSS.escape(chunkFile));
  const msg = document.getElementById('msg_' + CSS.escape(chunkFile));
  const card= document.getElementById('card_'+ CSS.escape(chunkFile));
  fetch('/save', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{chunk_file: chunkFile, text: ta.value}})
  }}).then(r => r.json()).then(() => {{
    msg.style.display = 'inline';
    setTimeout(() => msg.style.display = 'none', 2000);
    card.className = 'chunk corrected';
    card.querySelector('.badge').className = 'badge saved';
    card.querySelector('.badge').textContent = 'saved';
  }});
}}
function resetCorrection(chunkFile, geminiText) {{
  document.getElementById('ta_' + CSS.escape(chunkFile)).value = geminiText;
}}
function toggleMask(sourceFile) {{
  fetch('/mask', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{source_file: sourceFile}})
  }}).then(r => r.json()).then(d => {{
    const grp = document.getElementById('grp_' + CSS.escape(sourceFile));
    const hdr = document.getElementById('hdr_' + CSS.escape(sourceFile));
    const btn = document.getElementById('btn_' + CSS.escape(sourceFile));
    if (d.masked) {{
      grp.className = 'source-group masked';
      hdr.className = 'source-header masked';
      btn.className = 'btn-mask unmask';
      btn.textContent = 'Unmask';
    }} else {{
      grp.className = 'source-group active';
      hdr.className = 'source-header active';
      btn.className = 'btn-mask mask';
      btn.textContent = 'Mask';
    }}
  }});
}}
</script>
</body>
</html>"""

CHUNK_TEMPLATE = """
<div class="chunk {status_class}" id="card_{chunk_file}">
  <div class="chunk-header">
    <span class="chunk-id">{index}. {filename}</span>
    <span class="badge {badge_class}">{badge_text}</span>
  </div>
  <audio controls src="/audio/{filename}"></audio>
  <div class="transcripts">
    <div class="transcript-box">
      <label>Deepgram (raw)</label>
      <div class="raw-text">{raw_transcript}</div>
    </div>
    <div class="transcript-box">
      <label>Gemini corrected</label>
      <textarea id="ta_{chunk_file}" rows="5">{current_text}</textarea>
    </div>
  </div>
  <div class="actions">
    <button class="btn-save" onclick="saveCorrection('{chunk_file}', '{gemini_escaped}')">Save correction</button>
    <button class="btn-reset" onclick="resetCorrection('{chunk_file}', '{gemini_escaped}')">Reset to Gemini</button>
    <span class="save-msg" id="msg_{chunk_file}">&#10003; Saved</span>
  </div>
</div>
"""

DIAR_TEMPLATE = """
<div class="diar-section">
  <h2>Diarization segments — {source_file} ({n_kept} kept / {n_total} total)</h2>
  <table class="diar">
    <tr><th>#</th><th>Speaker</th><th>Start</th><th>End</th><th>Duration</th><th>Status</th></tr>
    {rows}
  </table>
</div>
"""

DIAR_ROW = """<tr class="{row_class}"><td>{i}</td><td>{speaker}</td><td>{start_s}s</td><td>{end_s}s</td><td>{duration_s}s</td><td>{status}</td></tr>"""


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI()


def _build_index() -> str:
    rows = _load_debug(OUTPUT_DIR)
    corrections = _load_corrections(OUTPUT_DIR)
    masked = _load_masked(OUTPUT_DIR)

    chunk_rows = [r for r in rows if r["type"] == "transcript_chunk"]

    by_source: dict[str, list] = {}
    for r in chunk_rows:
        by_source.setdefault(r["source_file"], []).append(r)

    rows_html = ""
    for i, (source_file, chunks) in enumerate(by_source.items(), 1):
        is_masked = source_file in masked
        n_corrected = sum(1 for c in chunks if c["chunk_file"] in corrections)
        row_class = "masked" if is_masked else ""
        if is_masked:
            badge = '<span class="badge masked">masked</span>'
        elif n_corrected:
            badge = f'<span class="badge corrected">{n_corrected} corrected</span>'
        else:
            badge = '<span class="badge ok">ok</span>'
        rows_html += INDEX_ROW.format(
            i=i,
            source_file=source_file,
            n_chunks=len(chunks),
            n_corrected=n_corrected,
            row_class=row_class,
            status_badge=badge,
        )

    n_corrected_total = sum(1 for r in chunk_rows if r["chunk_file"] in corrections)
    return INDEX_PAGE.format(
        output_dir=str(OUTPUT_DIR),
        n_sources=len(by_source),
        n_masked=len(masked),
        n_chunks=len(chunk_rows),
        n_corrected=n_corrected_total,
        histogram=_duration_histogram(chunk_rows),
        rows=rows_html,
    )


def _build_detail(source_file: str) -> str | None:
    rows = _load_debug(OUTPUT_DIR)
    corrections = _load_corrections(OUTPUT_DIR)
    masked = _load_masked(OUTPUT_DIR)

    chunk_rows = [r for r in rows if r["type"] == "transcript_chunk"]
    diar_rows  = [r for r in rows if r["type"] == "diarization_segment"]

    # Ordered list of sources for prev/next navigation
    sources = list(dict.fromkeys(r["source_file"] for r in chunk_rows))
    if source_file not in sources:
        return None
    idx = sources.index(source_file)
    prev_link = f'<a href="/source/{sources[idx-1]}">&#8592; prev</a>' if idx > 0 else ""
    next_link = f'<a href="/source/{sources[idx+1]}">next &#8594;</a>' if idx < len(sources) - 1 else ""

    chunks = [r for r in chunk_rows if r["source_file"] == source_file]
    diar_segs = [r for r in diar_rows if r["source_file"] == source_file]

    # Diarization section
    diarization_section = ""
    if diar_segs:
        n_kept = sum(1 for s in diar_segs if s["kept"] == "True")
        table_rows = "".join(
            DIAR_ROW.format(
                i=i + 1,
                speaker=s["speaker"],
                start_s=s["start_s"],
                end_s=s["end_s"],
                duration_s=s["duration_s"],
                status="kept" if s["kept"] == "True" else "discarded",
                row_class="main-speaker" if s["kept"] == "True" else "discarded",
            )
            for i, s in enumerate(diar_segs)
        )
        diarization_section = DIAR_TEMPLATE.format(
            source_file=source_file, n_kept=n_kept, n_total=len(diar_segs), rows=table_rows
        )

    is_masked = source_file in masked
    grp_class = "source-group masked" if is_masked else "source-group active"
    hdr_class  = "masked" if is_masked else "active"
    btn_class  = "unmask" if is_masked else "mask"
    btn_text   = "Unmask" if is_masked else "Mask"

    chunks_html = ""
    for i, row in enumerate(chunks, 1):
        chunk_file = row["chunk_file"]
        filename   = Path(chunk_file).name
        saved      = chunk_file in corrections
        current_text = corrections.get(chunk_file, row["corrected_transcript"])
        gemini_text  = row["corrected_transcript"]
        chunks_html += CHUNK_TEMPLATE.format(
            index=i,
            filename=filename,
            chunk_file=chunk_file,
            raw_transcript=row["raw_transcript"],
            current_text=current_text,
            gemini_escaped=gemini_text.replace("'", "\\'").replace("\n", "\\n"),
            status_class="corrected" if saved else "pending",
            badge_class="saved" if saved else "pending",
            badge_text="saved" if saved else "pending review",
        )

    n_corrected = sum(1 for c in chunks if c["chunk_file"] in corrections)
    return DETAIL_PAGE.format(
        source_file=source_file,
        n_chunks=len(chunks),
        n_corrected=n_corrected,
        grp_class=grp_class,
        hdr_class=hdr_class,
        btn_class=btn_class,
        btn_text=btn_text,
        chunks_html=chunks_html,
        diarization_section=diarization_section,
        prev_link=prev_link,
        next_link=next_link,
    )


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(_build_index())


@app.get("/source/{source_file:path}", response_class=HTMLResponse)
def source_detail(source_file: str):
    html = _build_detail(source_file)
    if html is None:
        return HTMLResponse("Source file not found", status_code=404)
    return HTMLResponse(html)


@app.post("/save")
async def save_correction(request: Request):
    data = await request.json()
    _save_correction(OUTPUT_DIR, data["chunk_file"], data["text"])
    return {"status": "ok"}


@app.post("/mask")
async def toggle_mask(request: Request):
    data = await request.json()
    now_masked = _toggle_mask(OUTPUT_DIR, data["source_file"])
    return {"masked": now_masked}


@app.get("/audio/{filename}")
def audio(filename: str):
    path = OUTPUT_DIR / "wavs" / filename
    if not path.exists():
        return HTMLResponse("not found", status_code=404)
    return FileResponse(path, media_type="audio/wav")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory produced by preprocess_audio.py (contains debug.csv and wavs/).",
)
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8765, show_default=True)
def main(output_dir: Path, host: str, port: int):
    global OUTPUT_DIR
    OUTPUT_DIR = output_dir
    print(f"Serving review UI at http://{host}:{port}  (output-dir: {output_dir})")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
