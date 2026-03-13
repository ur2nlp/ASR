"""Standalone WER/CER evaluation.

Loads a trained model and processor, runs inference on a test set, and
reports WER/CER. Supports optional LM decoding.

Usage:
    python -m tools.eval \
        --model_dir models/zulu/w2v300m_default/best-checkpoint \
        --test_data data/zulu/test \
        --lm_arpa path/to/lm.arpa
"""

import argparse
import json
import sys

import evaluate
import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCTC, AutoProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ASR model")
    parser.add_argument("--model_dir", required=True, help="Path to model checkpoint")
    parser.add_argument("--test_data", required=True, help="Path to test dataset on disk")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lm_arpa", default=None, help="Path to KenLM ARPA file")
    parser.add_argument("--lm_alpha", type=float, default=0.5)
    parser.add_argument("--lm_beta", type=float, default=1.5)
    parser.add_argument("--beam_width", type=int, default=100)
    parser.add_argument("--output", default=None, help="Path to write results JSON")
    return parser.parse_args()


def main():
    args = parse_args()

    processor = AutoProcessor.from_pretrained(args.model_dir)
    model = AutoModelForCTC.from_pretrained(args.model_dir)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataset = load_from_disk(args.test_data)

    # optionally set up LM decoding
    decode_with_lm = False
    if args.lm_arpa:
        from src.decoding import build_processor_with_lm
        processor_with_lm = build_processor_with_lm(
            processor,
            arpa_path=args.lm_arpa,
            alpha=args.lm_alpha,
            beta=args.lm_beta,
            beam_width=args.beam_width,
        )
        decode_with_lm = True

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    all_preds = []
    all_refs = []

    # detect input column
    if "input_features" in dataset.column_names:
        input_col = "input_features"
    else:
        input_col = "input_values"

    for i in tqdm(range(0, len(dataset), args.batch_size), desc="Evaluating"):
        batch = dataset[i:i + args.batch_size]

        inputs = processor.feature_extractor(
            batch[input_col],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits

        if decode_with_lm:
            from src.decoding import decode_logits
            preds = decode_logits(
                logits.cpu().numpy(),
                processor_with_lm,
                beam_width=args.beam_width,
            )
        else:
            pred_ids = logits.argmax(dim=-1)
            preds = processor.tokenizer.batch_decode(pred_ids, group_tokens=True)

        # get reference labels
        label_ids = batch["labels"]
        for label_seq in label_ids:
            filtered = [lid for lid in label_seq if lid != -100]
            ref = processor.tokenizer.decode(filtered, group_tokens=False)
            all_refs.append(ref)

        all_preds.extend(preds)

    # filter empty references
    pairs = [(p, r) for p, r in zip(all_preds, all_refs) if r.strip()]
    if not pairs:
        print("No valid references found.", file=sys.stderr)
        return

    preds_filtered, refs_filtered = zip(*pairs)

    wer = wer_metric.compute(predictions=list(preds_filtered), references=list(refs_filtered))
    cer = cer_metric.compute(predictions=list(preds_filtered), references=list(refs_filtered))

    results = {
        "wer": wer,
        "cer": cer,
        "num_examples": len(pairs),
        "model_dir": args.model_dir,
        "lm_arpa": args.lm_arpa,
    }

    print(f"WER: {wer:.4f}")
    print(f"CER: {cer:.4f}")
    print(f"Examples: {len(pairs)}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
