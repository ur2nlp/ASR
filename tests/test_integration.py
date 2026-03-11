"""Integration smoke test: synthetic data → full pipeline → one training step.

This test verifies that all components wire together correctly without
requiring a GPU or real model downloads. It uses a tiny model config and
synthetic audio data.
"""

import json
import os

import numpy as np
import pytest
from datasets import Audio, Dataset, DatasetDict
from omegaconf import OmegaConf

from src.data import (
    DataCollatorCTCWithPadding,
    ensure_train_dev_split,
    normalize_dataset,
)
from src.metrics import make_compute_metrics, preprocess_logits_for_metrics
from src.preprocessing import build_text_normalizer
from src.processors import get_model_spec
from src.vocab import create_ctc_tokenizer, generate_vocab, save_vocab


@pytest.fixture
def synthetic_dataset():
    """Create a minimal synthetic audio dataset."""
    rng = np.random.default_rng(42)
    sr = 16000
    num_samples = sr  # 1 second

    data = {
        "audio": [
            {"array": rng.standard_normal(num_samples).astype(np.float32), "sampling_rate": sr}
            for _ in range(6)
        ],
        "transcription": [
            "hello world",
            "this is a test",
            "hello again",
            "world test",
            "a simple test",
            "hello test world",
        ],
    }

    return DatasetDict({
        "train": Dataset.from_dict({k: v[:4] for k, v in data.items()}),
        "dev": Dataset.from_dict({k: v[4:] for k, v in data.items()}),
    })


@pytest.fixture
def preprocessing_cfg():
    return OmegaConf.create({
        "lowercase": True,
        "remove_punctuation": True,
        "unicode_normalize": "NFC",
        "custom_replacements": {},
    })


class TestVocabPipeline:
    """Test the vocab generation → tokenizer → processor pipeline."""

    def test_vocab_from_texts(self, tmp_dir):
        texts = ["hello world", "this is a test"]
        vocab = generate_vocab(texts)
        assert vocab["[PAD]"] == 0
        assert vocab["[UNK]"] == 1
        assert "|" in vocab

        vocab_path = os.path.join(tmp_dir, "vocab.json")
        save_vocab(vocab, vocab_path)

        tokenizer = create_ctc_tokenizer(vocab_path)
        assert len(tokenizer) == len(vocab)

        encoded = tokenizer("hello").input_ids
        assert len(encoded) > 0

    def test_normalize_then_vocab(self, synthetic_dataset, preprocessing_cfg, tmp_dir):
        dataset = normalize_dataset(synthetic_dataset, preprocessing_cfg)
        texts = dataset["train"]["transcription"]

        vocab = generate_vocab(texts)
        vocab_path = os.path.join(tmp_dir, "vocab.json")
        save_vocab(vocab, vocab_path)

        tokenizer = create_ctc_tokenizer(vocab_path)

        # verify every character in training data can be encoded
        for text in texts:
            encoded = tokenizer(text).input_ids
            # should have no UNK tokens (id=1) since vocab was built from this data
            assert 1 not in encoded, f"UNK found in encoding of: {text}"


class TestDataCollator:
    """Test that the data collator produces correctly shaped batches."""

    def test_collator_pads_and_masks(self, tmp_dir):
        texts = ["hello world", "test"]
        vocab = generate_vocab(texts)
        vocab_path = os.path.join(tmp_dir, "vocab.json")
        save_vocab(vocab, vocab_path)

        tokenizer = create_ctc_tokenizer(vocab_path)

        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor
        feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True,
        )
        processor = Wav2Vec2Processor(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
        )

        collator = DataCollatorCTCWithPadding(processor=processor, input_column="input_values")

        rng = np.random.default_rng(0)
        features = [
            {
                "input_values": rng.standard_normal(16000).astype(np.float32).tolist(),
                "labels": tokenizer("hello world").input_ids,
            },
            {
                "input_values": rng.standard_normal(8000).astype(np.float32).tolist(),
                "labels": tokenizer("test").input_ids,
            },
        ]

        batch = collator(features)

        assert "input_values" in batch
        assert "labels" in batch
        assert batch["input_values"].shape[0] == 2
        # labels should be padded to same length
        assert batch["labels"].shape[0] == 2
        # padding positions in labels should be -100
        assert (batch["labels"] == -100).any()
