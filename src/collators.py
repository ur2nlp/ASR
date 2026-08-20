"""Data collators for CTC and sequence-to-sequence ASR training.

These live in their own module (rather than in `data.py`) so that
`processors.py` can name them in its `ModelSpec` registry without importing
`data.py`, which itself imports from `processors.py`.

Both collators share the same contract: they receive pre-processed examples
carrying a model input column (`input_values` or `input_features`) and an
integer `labels` list, pad each with the appropriate semantics, and return a
batch dict ready for the model's forward pass.
"""

from dataclasses import dataclass

import torch
from transformers import ProcessorMixin


@dataclass
class DataCollatorCTCWithPadding:
    """Pad CTC inputs and labels with appropriate padding semantics.

    Inputs are padded with the feature extractor's padding value; labels are
    padded with -100 (ignored by CTC loss).

    Attributes:
        processor: The combined processor for padding.
        input_column: The name of the input feature column.
        padding: Padding strategy passed to the processor.
    """

    processor: ProcessorMixin
    input_column: str = "input_values"
    padding: bool | str = True

    def __call__(
        self,
        features: list[dict[str, list[int] | torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        # separate inputs and labels (different padding semantics)
        input_features = [
            {self.input_column: feature[self.input_column]} for feature in features
        ]
        # tokenizer.pad() expects dicts with key "input_ids", so we rename
        # from "labels" here and rename back after padding
        label_features = [
            {"input_ids": feature["labels"]} for feature in features
        ]

        # pad inputs
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # pad labels
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # replace padding tokens in labels with -100
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels

        return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Pad encoder-decoder ASR inputs and decoder labels.

    Differs from the CTC collator in one respect beyond naming: the tokenizer
    of a seq2seq ASR model prepends a decoder-start token (for Whisper, the
    `<|startoftranscript|>` token) to every encoded transcript. The model
    prepends that token itself when it shifts labels right to build
    `decoder_input_ids`, so leaving it in the labels would train the model to
    predict its own start token. We strip it here when every example in the
    batch begins with it.

    Attributes:
        processor: The combined processor for padding.
        input_column: The name of the input feature column.
        padding: Padding strategy passed to the processor.
        decoder_start_token_id: The model's decoder start token. When None, no
            leading token is stripped.
    """

    processor: ProcessorMixin
    input_column: str = "input_features"
    padding: bool | str = True
    decoder_start_token_id: int | None = None

    def __call__(
        self,
        features: list[dict[str, list[int] | torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        input_features = [
            {self.input_column: feature[self.input_column]} for feature in features
        ]
        label_features = [
            {"input_ids": feature["labels"]} for feature in features
        ]

        # Whisper's feature extractor already pads every clip to a fixed 30 s
        # window, so this is a no-op there; it still matters for seq2seq models
        # with variable-length inputs.
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt",
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # replace padding tokens with -100 so they are ignored by the loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # strip the decoder start token; the model re-adds it when shifting
        if self.decoder_start_token_id is not None and labels.shape[1] > 0:
            starts_with_bos = (labels[:, 0] == self.decoder_start_token_id).all()
            if starts_with_bos.item():
                labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
