"""Train a KenLM n-gram language model from text.

Wraps KenLM's `lmplz` command to build an ARPA file from training
transcriptions. The text can come from a HuggingFace dataset on disk
or a plain text file (one sentence per line).

Usage:
    python -m tools.train_lm \
        --text_source data/zulu/untokenized \
        --text_column transcription \
        --order 3 \
        --output lm/zulu_3gram.arpa
"""

import argparse
import os
import subprocess
import sys
import tempfile

from datasets import load_from_disk


def parse_args():
    parser = argparse.ArgumentParser(description="Train KenLM n-gram model")
    parser.add_argument(
        "--text_source", required=True,
        help="Path to HF dataset on disk or plain text file",
    )
    parser.add_argument("--text_column", default="transcription")
    parser.add_argument("--split", default="train")
    parser.add_argument("--order", type=int, default=3, help="N-gram order")
    parser.add_argument("--output", required=True, help="Output ARPA file path")
    parser.add_argument(
        "--kenlm_bin", default="lmplz",
        help="Path to KenLM lmplz binary",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # collect text
    if os.path.isdir(args.text_source):
        dataset = load_from_disk(args.text_source)
        if isinstance(dataset, dict):
            split = dataset[args.split]
        else:
            split = dataset
        texts = split[args.text_column]
    else:
        with open(args.text_source, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]

    print(f"Collected {len(texts)} sentences for LM training", file=sys.stderr)

    # write to temp file for lmplz
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
        for text in texts:
            tmp.write(text + "\n")
        tmp_path = tmp.name

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    try:
        cmd = [
            args.kenlm_bin,
            "-o", str(args.order),
            "--text", tmp_path,
            "--arpa", args.output,
        ]
        print(f"Running: {' '.join(cmd)}", file=sys.stderr)
        subprocess.run(cmd, check=True)
        print(f"ARPA model saved to {args.output}", file=sys.stderr)
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    main()
