"""Tests for src/data.py."""

import os

import numpy as np
import pytest
from datasets import Dataset, DatasetDict
from omegaconf import OmegaConf

from src.data import (
    DataCollatorCTCWithPadding,
    _standardize_columns,
    ensure_train_dev_split,
    normalize_dataset,
)


class TestStandardizeColumns:
    def test_rename_audio_column(self):
        ds = DatasetDict({
            "train": Dataset.from_dict({"sound": [1, 2], "text": ["a", "b"]}),
        })
        config = OmegaConf.create({"audio_column": "sound", "text_column": "text"})
        result = _standardize_columns(ds, config)
        assert "audio" in result["train"].column_names

    def test_rename_text_column(self):
        ds = DatasetDict({
            "train": Dataset.from_dict({"audio": [1, 2], "sentence": ["a", "b"]}),
        })
        config = OmegaConf.create({"audio_column": "audio", "text_column": "sentence"})
        result = _standardize_columns(ds, config)
        assert "transcription" in result["train"].column_names

    def test_no_rename_if_standard(self):
        ds = DatasetDict({
            "train": Dataset.from_dict({"audio": [1, 2], "transcription": ["a", "b"]}),
        })
        config = OmegaConf.create({"audio_column": "audio", "text_column": "transcription"})
        result = _standardize_columns(ds, config)
        assert "audio" in result["train"].column_names
        assert "transcription" in result["train"].column_names


class TestEnsureTrainDevSplit:
    def test_creates_dev_split(self):
        ds = DatasetDict({
            "train": Dataset.from_dict({"text": list(range(100))}),
        })
        result = ensure_train_dev_split(ds, dev_size=0.2)
        assert "dev" in result
        assert "train" in result
        assert len(result["dev"]) == 20

    def test_preserves_existing_dev(self):
        ds = DatasetDict({
            "train": Dataset.from_dict({"text": [1, 2, 3]}),
            "dev": Dataset.from_dict({"text": [4, 5]}),
        })
        result = ensure_train_dev_split(ds, dev_size=0.5)
        assert len(result["dev"]) == 2

    def test_preserves_test_split(self):
        ds = DatasetDict({
            "train": Dataset.from_dict({"text": list(range(100))}),
            "test": Dataset.from_dict({"text": [101, 102]}),
        })
        result = ensure_train_dev_split(ds, dev_size=10)
        assert "test" in result
        assert len(result["test"]) == 2

    def test_absolute_dev_size(self):
        ds = DatasetDict({
            "train": Dataset.from_dict({"text": list(range(100))}),
        })
        result = ensure_train_dev_split(ds, dev_size=15)
        assert len(result["dev"]) == 15


class TestNormalizeDataset:
    def test_applies_normalization(self, preprocessing_config):
        ds = DatasetDict({
            "train": Dataset.from_dict({"transcription": ["Hello, World!"]}),
        })
        result = normalize_dataset(ds, preprocessing_config)
        assert result["train"]["transcription"][0] == "hello world"

    def test_multiple_splits(self, preprocessing_config):
        ds = DatasetDict({
            "train": Dataset.from_dict({"transcription": ["Hello"]}),
            "test": Dataset.from_dict({"transcription": ["World!"]}),
        })
        result = normalize_dataset(ds, preprocessing_config)
        assert result["train"]["transcription"][0] == "hello"
        assert result["test"]["transcription"][0] == "world"
