"""Standalone WER/CER evaluation.

Loads a trained model and processor, runs inference on a test set, and
reports WER/CER. The architecture is recovered from the checkpoint's own
config, so CTC and seq2seq models are both handled: CTC models are decoded by
argmax (optionally with KenLM shallow fusion), seq2seq models autoregressively.

Usage:
    python -m tools.eval \
        --model_dir models/zulu/xlsr300m_l40_basic/best-checkpoint \
        --test_data data/zulu/test \
        --lm_arpa path/to/lm.arpa

    python -m tools.eval \
        --model_dir models/zulu/whisper_small_whisper_basic/best-checkpoint \
        --test_data data/zulu/test \
        --num_beams 5
"""

import argparse
import json
import sys

import evaluate
import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoProcessor

from src.processors import get_spec_for_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ASR model")
    parser.add_argument("--model_dir", required=True, help="Path to model checkpoint")
    parser.add_argument("--test_data", required=True, help="Path to test dataset on disk")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lm_arpa", default=None, help="Path to KenLM ARPA file (CTC only)")
    parser.add_argument("--lm_alpha", type=float, default=0.5)
    parser.add_argument("--lm_beta", type=float, default=1.5)
    parser.add_argument("--beam_width", type=int, default=100)
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="Beam size for autoregressive decoding (seq2seq only)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=225,
        help="Generation length cap (seq2seq only)",
    )
    parser.add_argument("--output", default=None, help="Path to write results JSON")
    return parser.parse_args()


def main():
    args = parse_args()

    model_type, spec = get_spec_for_checkpoint(args.model_dir)
    print(
        f"Checkpoint architecture: {model_type} (objective={spec.objective})",
        file=sys.stderr,
    )

    if args.lm_arpa and not spec.supports_lm_decoding:
        raise ValueError(
            f"--lm_arpa is not supported for model type '{model_type}'. KenLM "
            f"shallow fusion via pyctcdecode operates on per-frame CTC logits and "
            f"has no equivalent for an autoregressive decoder."
        )

    processor = AutoProcessor.from_pretrained(args.model_dir)
    model = spec.model_class.from_pretrained(args.model_dir)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataset = load_from_disk(args.test_data)

    # optionally set up LM decoding
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

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    all_preds = []
    all_refs = []

    input_column = spec.input_column
    if input_column not in dataset.column_names:
        raise ValueError(
            f"Test dataset has no '{input_column}' column (found: "
            f"{dataset.column_names}). It was prepared for a different "
            f"architecture than this checkpoint's."
        )

    for start in tqdm(range(0, len(dataset), args.batch_size), desc="Evaluating"):
        batch = dataset[start:start + args.batch_size]

        # the dataset already holds extracted features, so pad them into a
        # batch rather than re-running feature extraction over them
        inputs = processor.feature_extractor.pad(
            [{input_column: features} for features in batch[input_column]],
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            preds = _decode_batch(
                model=model,
                spec=spec,
                processor=processor,
                processor_with_lm=processor_with_lm,
                inputs=inputs,
                args=args,
            )

        # get reference labels
        for label_seq in batch["labels"]:
            filtered = [
                label_id for label_id in label_seq if label_id != -100
            ]
            reference = processor.tokenizer.decode(
                filtered, **spec.label_decode_kwargs
            )
            all_refs.append(reference)

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
        "model_type": model_type,
        "lm_arpa": args.lm_arpa,
    }

    print(f"WER: {wer:.4f}")
    print(f"CER: {cer:.4f}")
    print(f"Examples: {len(pairs)}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}", file=sys.stderr)


def _decode_batch(model, spec, processor, processor_with_lm, inputs, args) -> list[str]:
    """Decode one batch into transcript strings, per the spec's objective."""
    if spec.uses_generation:
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
        )
        return processor.tokenizer.batch_decode(
            generated_ids, **spec.prediction_decode_kwargs
        )

    logits = model(**inputs).logits

    if processor_with_lm is not None:
        from src.decoding import decode_logits
        return decode_logits(
            logits.cpu().numpy(),
            processor_with_lm,
            beam_width=args.beam_width,
        )

    pred_ids = logits.argmax(dim=-1)
    return processor.tokenizer.batch_decode(
        pred_ids, **spec.prediction_decode_kwargs
    )


if __name__ == "__main__":
    main()
