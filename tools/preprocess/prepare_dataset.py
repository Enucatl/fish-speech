#!/usr/bin/env python3
"""
Convert F5-TTS barbero dataset to fish-speech format.

Input:  pipe-separated CSV with columns: audio_file|text
Output: data/barbero/ with .wav files (symlinked) and .lab files
"""

import csv
import os
import sys
from pathlib import Path

SPLITS = {
    "barbero": Path("/home/user/data/barbero/metadata.csv"),
    "barbero-test": Path("/home/user/data/barbero-test/metadata.csv"),
}

OUT_BASE = Path(__file__).parent / "data" / "barbero"
OUT_BASE.mkdir(parents=True, exist_ok=True)

total = 0
skipped = 0

for split_name, csv_path in SPLITS.items():
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            wav_path = Path(row["audio_file"])
            text = row["text"].strip().strip('"')

            if not wav_path.exists():
                print(f"WARN: missing {wav_path}", file=sys.stderr)
                skipped += 1
                continue

            stem = wav_path.stem
            out_wav = OUT_BASE / wav_path.name
            out_lab = OUT_BASE / (stem + ".lab")

            # Symlink the wav to avoid duplicating large files
            if not out_wav.exists():
                out_wav.symlink_to(wav_path.resolve())

            out_lab.write_text(text, encoding="utf-8")
            total += 1

print(f"Done: {total} samples written to {OUT_BASE}  ({skipped} skipped)")
