"""Tests for src/data.py."""

import os

import numpy as np
import pytest
import soundfile as sf
from datasets import Audio, Dataset, DatasetDict
from omegaconf import OmegaConf

from src.data import (
    DataCollatorCTCWithPadding,
    _load_paired,
    _standardize_columns,
    ensure_train_dev_split,
    load_dataset_from_config,
    load_external_eval_sets,
    normalize_dataset,
)
from src.processors import setup_processor
from src.vocab import create_ctc_tokenizer


@pytest.fixture
def wav2vec2_processor(base_config, vocab_path):
    """A cheap, network-free wav2vec2 processor built from the sample vocab."""
    tokenizer = create_ctc_tokenizer(vocab_path)
    return setup_processor(base_config, tokenizer)


def _write_pair(directory: str, stem: str, transcription: str, sr: int = 16000):
    """Write a synthetic .wav / .txt pair under `directory`."""
    os.makedirs(directory, exist_ok=True)
    audio = np.zeros(sr // 10, dtype=np.float32)
    sf.write(os.path.join(directory, f"{stem}.wav"), audio, sr)
    with open(os.path.join(directory, f"{stem}.txt"), "w", encoding="utf-8") as f:
        f.write(transcription)


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


class TestLoadPaired:
    def test_flat_directory(self, tmp_dir):
        _write_pair(tmp_dir, "utt001", "hello world")
        _write_pair(tmp_dir, "utt002", "goodbye")
        config = OmegaConf.create({"type": "paired", "path": tmp_dir})

        result = _load_paired(config)

        assert set(result.keys()) == {"train"}
        assert len(result["train"]) == 2
        assert set(result["train"].column_names) == {"audio", "transcription"}
        assert isinstance(result["train"].features["audio"], Audio)
        # sorted by filename, so utt001 comes first
        assert result["train"]["transcription"][0] == "hello world"

    def test_strips_transcription_whitespace(self, tmp_dir):
        _write_pair(tmp_dir, "utt001", "  hello world\n")
        config = OmegaConf.create({"type": "paired", "path": tmp_dir})

        result = _load_paired(config)

        assert result["train"]["transcription"][0] == "hello world"

    def test_presplit_subdirectories(self, tmp_dir):
        _write_pair(os.path.join(tmp_dir, "train"), "a", "train utterance")
        _write_pair(os.path.join(tmp_dir, "dev"), "b", "dev utterance")
        _write_pair(os.path.join(tmp_dir, "test"), "c", "test utterance")
        config = OmegaConf.create({"type": "paired", "path": tmp_dir})

        result = _load_paired(config)

        assert set(result.keys()) == {"train", "dev", "test"}
        assert result["dev"]["transcription"][0] == "dev utterance"

    def test_recursive(self, tmp_dir):
        _write_pair(os.path.join(tmp_dir, "spk1"), "a", "one")
        _write_pair(os.path.join(tmp_dir, "spk2"), "b", "two")
        config = OmegaConf.create(
            {"type": "paired", "path": tmp_dir, "recursive": True}
        )

        result = _load_paired(config)

        assert len(result["train"]) == 2

    def test_non_recursive_ignores_subdirs(self, tmp_dir):
        _write_pair(tmp_dir, "top", "top level")
        _write_pair(os.path.join(tmp_dir, "nested"), "deep", "nested")
        config = OmegaConf.create({"type": "paired", "path": tmp_dir})

        result = _load_paired(config)

        assert len(result["train"]) == 1

    def test_skips_audio_without_transcript(self, tmp_dir):
        _write_pair(tmp_dir, "utt001", "has transcript")
        # audio with no matching .txt
        sf.write(os.path.join(tmp_dir, "utt002.wav"), np.zeros(1600, np.float32), 16000)
        config = OmegaConf.create({"type": "paired", "path": tmp_dir})

        result = _load_paired(config)

        assert len(result["train"]) == 1

    def test_custom_extension(self, tmp_dir):
        audio = np.zeros(1600, dtype=np.float32)
        sf.write(os.path.join(tmp_dir, "utt001.flac"), audio, 16000)
        with open(os.path.join(tmp_dir, "utt001.txt"), "w") as f:
            f.write("flac audio")
        config = OmegaConf.create(
            {"type": "paired", "path": tmp_dir, "audio_ext": ".flac"}
        )

        result = _load_paired(config)

        assert len(result["train"]) == 1

    def test_missing_directory_raises(self, tmp_dir):
        config = OmegaConf.create(
            {"type": "paired", "path": os.path.join(tmp_dir, "nonexistent")}
        )
        with pytest.raises(FileNotFoundError):
            _load_paired(config)

    def test_no_audio_files_raises(self, tmp_dir):
        config = OmegaConf.create({"type": "paired", "path": tmp_dir})
        with pytest.raises(FileNotFoundError):
            _load_paired(config)

    def test_dispatcher_routes_and_caches(self, tmp_dir):
        data_dir = os.path.join(tmp_dir, "raw")
        _write_pair(data_dir, "utt001", "hello world")
        cache_dir = os.path.join(tmp_dir, "cache")
        config = OmegaConf.create({"type": "paired", "path": data_dir})

        result = load_dataset_from_config(config, cache_dir)
        assert len(result["train"]) == 1
        # second call should load from the on-disk cache
        assert os.path.isdir(os.path.join(cache_dir, "untokenized"))
        cached = load_dataset_from_config(config, cache_dir)
        assert len(cached["train"]) == 1


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


class TestLoadExternalEvalSets:
    def test_returns_dict_of_processed_sets(
        self, tmp_dir, preprocessing_config, wav2vec2_processor
    ):
        set_a = os.path.join(tmp_dir, "clean")
        set_b = os.path.join(tmp_dir, "noisy")
        _write_pair(set_a, "a", "hello world")
        _write_pair(set_b, "b", "test")
        configs = [
            {"name": "clean", "path": set_a},
            {"name": "noisy", "path": set_b},
        ]

        result = load_external_eval_sets(
            configs,
            preprocessing_config=preprocessing_config,
            processor=wav2vec2_processor,
            model_type="wav2vec2",
        )

        assert set(result.keys()) == {"clean", "noisy"}
        assert set(result["clean"].column_names) == {
            "input_values",
            "labels",
            "input_length",
        }
        assert len(result["clean"]) == 1
        # input_length must match the feature length for group_by_length batching
        example = result["clean"][0]
        assert example["input_length"] == len(example["input_values"])

    def test_normalizes_transcriptions(
        self, tmp_dir, preprocessing_config, wav2vec2_processor
    ):
        set_dir = os.path.join(tmp_dir, "set")
        _write_pair(set_dir, "a", "Hello, World!")

        result = load_external_eval_sets(
            [{"name": "x", "path": set_dir}],
            preprocessing_config=preprocessing_config,
            processor=wav2vec2_processor,
            model_type="wav2vec2",
        )

        # transcription should be normalized before label encoding
        expected = wav2vec2_processor.tokenizer("hello world").input_ids
        assert result["x"][0]["labels"] == expected

    def test_unsupported_type_raises(
        self, tmp_dir, preprocessing_config, wav2vec2_processor
    ):
        _write_pair(tmp_dir, "a", "hello")
        with pytest.raises(ValueError, match="unsupported type"):
            load_external_eval_sets(
                [{"name": "x", "path": tmp_dir, "type": "huggingface"}],
                preprocessing_config=preprocessing_config,
                processor=wav2vec2_processor,
                model_type="wav2vec2",
            )

    def test_duplicate_name_raises(
        self, tmp_dir, preprocessing_config, wav2vec2_processor
    ):
        set_a = os.path.join(tmp_dir, "a")
        set_b = os.path.join(tmp_dir, "b")
        _write_pair(set_a, "x", "hello")
        _write_pair(set_b, "y", "world")
        with pytest.raises(ValueError, match="Duplicate"):
            load_external_eval_sets(
                [{"name": "dup", "path": set_a}, {"name": "dup", "path": set_b}],
                preprocessing_config=preprocessing_config,
                processor=wav2vec2_processor,
                model_type="wav2vec2",
            )
