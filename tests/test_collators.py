"""Tests for src/collators.py.

The seq2seq collator is exercised through the network-free wav2vec2 processor
fixture rather than a real Whisper processor: both collators depend only on the
feature-extractor/tokenizer padding contract, not on any Whisper-specific
behaviour, and the logic worth testing here is label padding and the stripping
of the decoder start token.
"""

import pytest

from src.collators import (
    DataCollatorCTCWithPadding,
    DataCollatorSpeechSeq2SeqWithPadding,
)
from src.processors import setup_processor
from src.vocab import create_ctc_tokenizer


DECODER_START_TOKEN_ID = 3


@pytest.fixture
def processor(base_config, vocab_path):
    """A cheap, network-free processor providing the padding contract."""
    tokenizer = create_ctc_tokenizer(vocab_path)
    return setup_processor(base_config, tokenizer)


def _features(label_lists, input_lengths):
    """Build collator input features with the given labels and input lengths."""
    return [
        {"input_values": [0.0] * length, "labels": labels}
        for labels, length in zip(label_lists, input_lengths)
    ]


class TestCTCCollator:
    def test_pads_labels_with_negative_hundred(self, processor):
        collator = DataCollatorCTCWithPadding(processor=processor)
        batch = collator(_features([[3, 4, 5], [6]], [16, 8]))

        assert batch["labels"].shape == (2, 3)
        assert batch["labels"][1].tolist() == [6, -100, -100]

    def test_pads_inputs_to_longest(self, processor):
        collator = DataCollatorCTCWithPadding(processor=processor)
        batch = collator(_features([[3], [4]], [16, 8]))

        assert batch["input_values"].shape == (2, 16)


class TestSeq2SeqCollator:
    def test_strips_decoder_start_token_when_present_on_all(self, processor):
        collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=processor,
            input_column="input_values",
            decoder_start_token_id=DECODER_START_TOKEN_ID,
        )
        batch = collator(
            _features([[DECODER_START_TOKEN_ID, 7, 8], [DECODER_START_TOKEN_ID, 9]], [16, 8])
        )

        # leading start token removed from every row
        assert batch["labels"].shape == (2, 2)
        assert batch["labels"][0].tolist() == [7, 8]
        assert batch["labels"][1].tolist() == [9, -100]

    def test_keeps_leading_token_when_not_uniform(self, processor):
        collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=processor,
            input_column="input_values",
            decoder_start_token_id=DECODER_START_TOKEN_ID,
        )
        batch = collator(_features([[DECODER_START_TOKEN_ID, 7], [9, 8]], [16, 8]))

        assert batch["labels"].shape == (2, 2)
        assert batch["labels"][0].tolist() == [DECODER_START_TOKEN_ID, 7]

    def test_no_stripping_without_start_token_id(self, processor):
        collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=processor,
            input_column="input_values",
            decoder_start_token_id=None,
        )
        batch = collator(_features([[DECODER_START_TOKEN_ID, 7]], [16]))

        assert batch["labels"][0].tolist() == [DECODER_START_TOKEN_ID, 7]

    def test_pads_labels_with_negative_hundred(self, processor):
        collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=processor,
            input_column="input_values",
            decoder_start_token_id=None,
        )
        batch = collator(_features([[7, 8, 9], [4]], [16, 8]))

        assert batch["labels"][1].tolist() == [4, -100, -100]
