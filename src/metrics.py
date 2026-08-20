"""WER/CER computation and logits preprocessing.

Provides a closure-based compute_metrics factory (capturing the processor and
the architecture's decoding conventions) and a preprocess_logits_for_metrics
function that reduces logits via argmax before they accumulate in memory
during evaluation.

The two objectives decode differently. CTC predictions are per-frame token IDs
that must be collapsed across repeats and blanks (`group_tokens=True`), while
references are already one ID per character (`group_tokens=False`). Seq2seq
predictions and references are both subword sequences wrapped in special
tokens, which are stripped from each (`skip_special_tokens=True`). The caller
supplies these kwargs from the ModelSpec rather than this module branching on
model type.
"""

import torch
import evaluate


# Default decoding conventions, preserved for CTC callers that do not pass
# explicit kwargs.
DEFAULT_PREDICTION_DECODE_KWARGS = {"group_tokens": True}
DEFAULT_LABEL_DECODE_KWARGS = {"group_tokens": False}


def make_compute_metrics(
    processor,
    prediction_decode_kwargs: dict | None = None,
    label_decode_kwargs: dict | None = None,
):
    """Build a compute_metrics closure for the HF Trainer.

    Args:
        processor: The combined processor whose tokenizer is used to decode
            predicted and reference label IDs.
        prediction_decode_kwargs: Extra kwargs passed to `batch_decode` for
            model predictions. Defaults to the CTC convention.
        label_decode_kwargs: Extra kwargs passed to `batch_decode` for
            reference labels. Defaults to the CTC convention.

    Returns:
        A callable compatible with Trainer's compute_metrics argument.
    """
    if prediction_decode_kwargs is None:
        prediction_decode_kwargs = DEFAULT_PREDICTION_DECODE_KWARGS
    if label_decode_kwargs is None:
        label_decode_kwargs = DEFAULT_LABEL_DECODE_KWARGS

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        pad_token_id = processor.tokenizer.pad_token_id

        # replace -100 with pad token id for decoding. The Trainer pads
        # generated sequences with -100 when batches differ in length, so
        # predictions need the same treatment as labels under generation.
        label_ids[label_ids == -100] = pad_token_id
        pred_ids[pred_ids == -100] = pad_token_id

        pred_str = processor.tokenizer.batch_decode(
            pred_ids, **prediction_decode_kwargs
        )
        label_str = processor.tokenizer.batch_decode(
            label_ids, **label_decode_kwargs
        )

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

    Only applies to models scored from raw logits. Under
    `predict_with_generate` the Trainer already hands back token IDs, so this
    must not be installed for generation-based evaluation.

    Args:
        logits: Raw model output logits. (batch_size, seq_len, vocab_size)
        labels: Label tensor (unused, required by Trainer signature).

    Returns:
        Predicted token IDs. (batch_size, seq_len)
    """
    return logits.argmax(dim=-1)
