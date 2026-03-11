"""WER/CER computation and logits preprocessing for CTC models.

Provides a closure-based compute_metrics factory (capturing the processor)
and a preprocess_logits_for_metrics function that reduces logits via argmax
before they accumulate in memory during evaluation.
"""

import torch
import evaluate


def make_compute_metrics(processor):
    """Build a compute_metrics closure for the HF Trainer.

    Args:
        processor: The combined processor whose tokenizer is used to decode
            predicted and reference label IDs.

    Returns:
        A callable compatible with Trainer's compute_metrics argument.
    """
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with pad token id for decoding
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, group_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, group_tokens=False)

        # filter out empty references to avoid division by zero
        pairs = [
            (p, l) for p, l in zip(pred_str, label_str) if l.strip()
        ]
        if not pairs:
            return {"wer": 1.0, "cer": 1.0}

        filtered_pred, filtered_label = zip(*pairs)

        wer = wer_metric.compute(
            predictions=list(filtered_pred),
            references=list(filtered_label),
        )
        cer = cer_metric.compute(
            predictions=list(filtered_pred),
            references=list(filtered_label),
        )

        return {"wer": wer, "cer": cer}

    return compute_metrics


def preprocess_logits_for_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Reduce logits to predicted IDs to prevent OOM during evaluation.

    The Trainer accumulates predictions across the full eval set. Storing
    full logits (batch, seq_len, vocab_size) would exhaust memory for long
    sequences. Argmax reduces this to (batch, seq_len).

    Args:
        logits: Raw model output logits. (batch_size, seq_len, vocab_size)
        labels: Label tensor (unused, required by Trainer signature).

    Returns:
        Predicted token IDs. (batch_size, seq_len)
    """
    return logits.argmax(dim=-1)
