#!/usr/bin/env python3
"""
Convert bmo_voice_fetch output (dialogue.csv + WAVs) into a Piper TTS dataset.

Piper expects:
  - A directory with wav/ and metadata.csv
  - metadata.csv: pipe-delimited, no header, format "id|text" (id = filename stem)
  - WAVs: mono, 22050 Hz (high/medium quality)

Usage:
  python prepare_piper_dataset.py wav_output/dialogue.csv -o piper_dataset
  python prepare_piper_dataset.py wav_output/dialogue.csv -o piper_dataset --no-normalize  # skip audio conversion
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

# Optional: normalize audio to 22050 Hz mono
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False


def normalize_audio(src: Path, dst: Path, sample_rate: int = 22050) -> None:
    """Convert WAV to mono and resample to sample_rate."""
    if not PYDUB_AVAILABLE:
        raise RuntimeError("pydub required for audio normalization; run: pip install pydub")
    dst.parent.mkdir(parents=True, exist_ok=True)
    audio = AudioSegment.from_wav(str(src))
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(sample_rate)
    audio.export(str(dst), format="wav")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert dialogue CSV + WAVs into Piper TTS dataset format."
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to dialogue.csv (from bmo_voice_fetch.py)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("piper_dataset"),
        help="Output dataset directory (default: piper_dataset)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Target sample rate for WAVs (default: 22050 for Piper high/medium)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Copy WAVs as-is; do not convert to mono/22050 Hz (use if already correct)",
    )
    args = parser.parse_args()

    csv_path = args.csv_path
    if not csv_path.is_file():
        print(f"Error: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    # Resolve paths relative to CSV's directory so wav_path in CSV works
    base_dir = csv_path.resolve().parent
    out_dir = args.output_dir.resolve()
    wav_dir = out_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wav_path_str = row.get("wav_path", "").strip()
            dialogue = row.get("dialogue", "").strip()
            if not wav_path_str or dialogue is None:
                continue
            # Path may be relative to CWD or to CSV dir
            wav_path = Path(wav_path_str)
            if not wav_path.is_absolute():
                wav_path = base_dir / wav_path
            if not wav_path.exists():
                print(f"Warning: WAV not found, skipping: {wav_path}", file=sys.stderr)
                continue
            text = dialogue.replace("|", " ")  # Piper uses | as delimiter
            rows.append((wav_path, text))

    if not rows:
        print("No valid rows found in CSV.", file=sys.stderr)
        sys.exit(1)

    normalize = not args.no_normalize
    if normalize and not PYDUB_AVAILABLE:
        print("Warning: pydub not installed; copying WAVs as-is. Install pydub for 22050 Hz mono.", file=sys.stderr)
        normalize = False

    for wav_path, text in rows:
        stem = wav_path.stem
        dest = wav_dir / wav_path.name
        if normalize:
            try:
                normalize_audio(wav_path, dest, sample_rate=args.sample_rate)
            except Exception as e:
                print(f"Warning: could not normalize {wav_path.name}: {e}", file=sys.stderr)
                shutil.copy2(wav_path, dest)
        else:
            shutil.copy2(wav_path, dest)

    # Piper metadata: id|text (id = filename without extension), no header
    metadata_path = out_dir / "metadata.csv"
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        for wav_path, text in rows:
            stem = wav_path.stem
            # Escape any newlines in text for single line per utterance
            line = f"{stem}|{text.replace(chr(10), ' ').replace(chr(13), ' ')}\n"
            f.write(line)

    print(f"Piper dataset written to: {out_dir}")
    print(f"  wav/: {len(rows)} files")
    print(f"  metadata.csv: pipe-delimited id|text (no header)")
    print()
    print("Next steps (Piper training):")
    print("  1. Install Piper training env: https://github.com/rhasspy/piper")
    print("  2. Preprocess:")
    print(f"     python3 -m piper_train.preprocess \\")
    print(f"       --language en-us \\")
    print(f"       --input-dir {out_dir} \\")
    print(f"       --output-dir {out_dir}/training \\")
    print(f"       --dataset-format ljspeech \\")
    print(f"       --single-speaker \\")
    print(f"       --sample-rate {args.sample_rate}")
    print("  3. Fine-tune from a checkpoint (see PIPER_TRAINING.md)")


if __name__ == "__main__":
    main()
