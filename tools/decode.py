"""Batch inference on audio files, writing results to JSONL.

Usage:
    python -m tools.decode \
        --model_dir models/zulu/w2v300m_default/best-checkpoint \
        --audio_dir data/test_audio/ \
        --output predictions.jsonl
"""

import argparse
import glob
import json
import os
import sys

import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoModelForCTC, AutoProcessor


def _require_ctc_checkpoint(model_dir: str) -> None:
    """Fail fast on a seq2seq checkpoint rather than mis-decoding it.

    This tool decodes by argmax over per-frame logits, which is meaningless for
    an autoregressive decoder. Use `tools/eval.py`, which dispatches on the
    checkpoint's objective, until batch generation lands here too.
    """
    from src.processors import get_spec_for_checkpoint

    model_type, spec = get_spec_for_checkpoint(model_dir)
    if spec.objective != "ctc":
        raise ValueError(
            f"{__file__} supports CTC checkpoints only, but {model_dir} is a "
            f"'{model_type}' model with a '{spec.objective}' objective. "
            f"Use `python -m tools.eval` for WER/CER on this checkpoint."
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Batch decode audio files")
    parser.add_argument("--model_dir", required=True, help="Path to model checkpoint")
    parser.add_argument("--audio_dir", required=True, help="Directory of audio files")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--sampling_rate", type=int, default=16000)
    parser.add_argument("--lm_arpa", default=None, help="Path to KenLM ARPA file")
    parser.add_argument("--lm_alpha", type=float, default=0.5)
    parser.add_argument("--lm_beta", type=float, default=1.5)
    parser.add_argument("--beam_width", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()

    _require_ctc_checkpoint(args.model_dir)

    processor = AutoProcessor.from_pretrained(args.model_dir)
    model = AutoModelForCTC.from_pretrained(args.model_dir)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # set up LM decoding if requested
    processor_with_lm = None
    if args.lm_arpa:
        from src.decoding import build_processor_with_lm
        processor_with_lm = build_processor_with_lm(
            processor,
            arpa_path=args.lm_arpa,
            alpha=args.lm_alpha,
            beta=args.lm_beta,
            beam_width=args.beam_width,
        )

    # find audio files
    audio_extensions = ("*.wav", "*.flac", "*.mp3", "*.ogg")
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(args.audio_dir, "**", ext), recursive=True))
    audio_files.sort()

    if not audio_files:
        print(f"No audio files found in {args.audio_dir}", file=sys.stderr)
        return

    print(f"Found {len(audio_files)} audio files", file=sys.stderr)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as out_f:
        for audio_path in tqdm(audio_files, desc="Decoding"):
            audio_array, sr = sf.read(audio_path)

            inputs = processor.feature_extractor(
                audio_array,
                sampling_rate=args.sampling_rate,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = model(**inputs).logits

            if processor_with_lm is not None:
                from src.decoding import decode_logits
                transcription = decode_logits(
                    logits.cpu().numpy(),
                    processor_with_lm,
                    beam_width=args.beam_width,
                )[0]
            else:
                pred_ids = logits.argmax(dim=-1)
                transcription = processor.tokenizer.batch_decode(
                    pred_ids, group_tokens=True
                )[0]

            result = {
                "file": os.path.relpath(audio_path, args.audio_dir),
                "transcription": transcription,
            }
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Wrote {len(audio_files)} predictions to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
