"""Tests for src/data.py."""

import os

import numpy as np
import pytest
import soundfile as sf
from datasets import Audio, Dataset, DatasetDict
from omegaconf import OmegaConf

from src.data import (
    DataCollatorCTCWithPadding,
    assemble_splits,
    build_feature_cache,
    default_assignment,
    encode_labels,
    load_dataset_from_config,
    load_external_eval_sets,
    load_feature_cache,
    normalize_texts,
)
from src.sources.base import standardize_columns
from src.sources.paired import PairedDataset
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
        result = standardize_columns(ds, "sound", "text")
        assert "audio" in result["train"].column_names

    def test_rename_text_column(self):
        ds = DatasetDict({
            "train": Dataset.from_dict({"audio": [1, 2], "sentence": ["a", "b"]}),
        })
        result = standardize_columns(ds, "audio", "sentence")
        assert "transcription" in result["train"].column_names

    def test_no_rename_if_standard(self):
        ds = DatasetDict({
            "train": Dataset.from_dict({"audio": [1, 2], "transcription": ["a", "b"]}),
        })
        result = standardize_columns(ds, "audio", "transcription")
        assert "audio" in result["train"].column_names
        assert "transcription" in result["train"].column_names


class TestDefaultAssignment:
    def test_creates_dev_split(self):
        result = default_assignment(["train"] * 100, dev_size=0.2)
        assert len(result["dev"]) == 20
        assert len(result["train"]) == 80
        assert set(result["train"]).isdisjoint(result["dev"])

    def test_matches_train_test_split(self):
        """Same partition the old split produced, so a cached vocab stays valid."""
        old = Dataset.from_dict({"text": list(range(100))}).train_test_split(
            test_size=0.2, seed=1
        )
        result = default_assignment(["train"] * 100, dev_size=0.2, seed=1)
        assert sorted(result["dev"]) == sorted(old["test"]["text"])

    def test_preserves_existing_dev(self):
        result = default_assignment(["train"] * 3 + ["dev"] * 2, dev_size=0.5)
        assert result["dev"] == [3, 4]
        assert result["train"] == [0, 1, 2]

    def test_preserves_test_split(self):
        result = default_assignment(["train"] * 100 + ["test"] * 2, dev_size=10)
        assert result["test"] == [100, 101]
        assert len(result["dev"]) == 10

    def test_absolute_dev_size(self):
        result = default_assignment(["train"] * 100, dev_size=15)
        assert len(result["dev"]) == 15


class TestLoadPaired:
    def test_flat_directory(self, tmp_dir):
        _write_pair(tmp_dir, "utt001", "hello world")
        _write_pair(tmp_dir, "utt002", "goodbye")
        config = OmegaConf.create({"type": "paired", "path": tmp_dir})

        result = PairedDataset.from_config("", OmegaConf.to_container(config, resolve=True)).load()

        assert set(result.keys()) == {"train"}
        assert len(result["train"]) == 2
        assert set(result["train"].column_names) == {"audio", "transcription"}
        assert isinstance(result["train"].features["audio"], Audio)
        # sorted by filename, so utt001 comes first
        assert result["train"]["transcription"][0] == "hello world"

    def test_strips_transcription_whitespace(self, tmp_dir):
        _write_pair(tmp_dir, "utt001", "  hello world\n")
        config = OmegaConf.create({"type": "paired", "path": tmp_dir})

        result = PairedDataset.from_config("", OmegaConf.to_container(config, resolve=True)).load()

        assert result["train"]["transcription"][0] == "hello world"

    def test_presplit_subdirectories(self, tmp_dir):
        _write_pair(os.path.join(tmp_dir, "train"), "a", "train utterance")
        _write_pair(os.path.join(tmp_dir, "dev"), "b", "dev utterance")
        _write_pair(os.path.join(tmp_dir, "test"), "c", "test utterance")
        config = OmegaConf.create({"type": "paired", "path": tmp_dir})

        result = PairedDataset.from_config("", OmegaConf.to_container(config, resolve=True)).load()

        assert set(result.keys()) == {"train", "dev", "test"}
        assert result["dev"]["transcription"][0] == "dev utterance"

    def test_recursive(self, tmp_dir):
        _write_pair(os.path.join(tmp_dir, "spk1"), "a", "one")
        _write_pair(os.path.join(tmp_dir, "spk2"), "b", "two")
        config = OmegaConf.create(
            {"type": "paired", "path": tmp_dir, "recursive": True}
        )

        result = PairedDataset.from_config("", OmegaConf.to_container(config, resolve=True)).load()

        assert len(result["train"]) == 2

    def test_non_recursive_ignores_subdirs(self, tmp_dir):
        _write_pair(tmp_dir, "top", "top level")
        _write_pair(os.path.join(tmp_dir, "nested"), "deep", "nested")
        config = OmegaConf.create({"type": "paired", "path": tmp_dir})

        result = PairedDataset.from_config("", OmegaConf.to_container(config, resolve=True)).load()

        assert len(result["train"]) == 1

    def test_skips_audio_without_transcript(self, tmp_dir):
        _write_pair(tmp_dir, "utt001", "has transcript")
        # audio with no matching .txt
        sf.write(os.path.join(tmp_dir, "utt002.wav"), np.zeros(1600, np.float32), 16000)
        config = OmegaConf.create({"type": "paired", "path": tmp_dir})

        result = PairedDataset.from_config("", OmegaConf.to_container(config, resolve=True)).load()

        assert len(result["train"]) == 1

    def test_custom_extension(self, tmp_dir):
        audio = np.zeros(1600, dtype=np.float32)
        sf.write(os.path.join(tmp_dir, "utt001.flac"), audio, 16000)
        with open(os.path.join(tmp_dir, "utt001.txt"), "w") as f:
            f.write("flac audio")
        config = OmegaConf.create(
            {"type": "paired", "path": tmp_dir, "audio_ext": ".flac"}
        )

        result = PairedDataset.from_config("", OmegaConf.to_container(config, resolve=True)).load()

        assert len(result["train"]) == 1

    def test_missing_directory_raises(self, tmp_dir):
        config = OmegaConf.create(
            {"type": "paired", "path": os.path.join(tmp_dir, "nonexistent")}
        )
        with pytest.raises(FileNotFoundError):
            PairedDataset.from_config("", OmegaConf.to_container(config, resolve=True)).load()

    def test_no_audio_files_raises(self, tmp_dir):
        config = OmegaConf.create({"type": "paired", "path": tmp_dir})
        with pytest.raises(FileNotFoundError):
            PairedDataset.from_config("", OmegaConf.to_container(config, resolve=True)).load()

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


class TestNormalizeTexts:
    def test_applies_normalization(self, preprocessing_config):
        assert normalize_texts(["Hello, World!"], preprocessing_config) == ["hello world"]

    def test_preserves_order(self, preprocessing_config):
        assert normalize_texts(["Hello", "World!"], preprocessing_config) == ["hello", "world"]


class TestFeatureCachePipeline:
    """Features are extracted once; labels, splits and filters apply by index."""

    @pytest.fixture
    def corpus(self, tmp_dir):
        data_dir = os.path.join(tmp_dir, "data")
        _write_pair(data_dir, "rec1_000", "hello")
        _write_pair(data_dir, "rec1_001", "world")
        _write_pair(data_dir, "rec2_000", "hello world")
        config = OmegaConf.create({"type": "paired", "path": data_dir})
        return load_dataset_from_config(config, os.path.join(tmp_dir, "cache"))

    @pytest.fixture
    def features(self, corpus, tmp_dir, wav2vec2_processor):
        path = os.path.join(tmp_dir, "cache", "features")
        build_feature_cache(
            corpus,
            wav2vec2_processor.feature_extractor,
            input_column="input_values",
            sampling_rate=16000,
            cache_path=path,
        )
        return load_feature_cache(path, ["train"])

    def test_cache_holds_every_example_and_its_basename(self, features):
        assert len(features) == 3
        assert features["audio_path"] == ["rec1_000.wav", "rec1_001.wav", "rec2_000.wav"]
        assert features["source_split"] == ["train"] * 3
        assert features["transcription"] == ["hello", "world", "hello world"]
        assert features["input_length"] == [1600] * 3

    def test_nothing_is_written_beside_the_source(self, corpus, features, tmp_dir):
        untokenized = os.path.join(tmp_dir, "cache", "untokenized", "train")
        assert not any(name.startswith("cache-") for name in os.listdir(untokenized))

    def test_no_partial_file_left_behind(self, features, tmp_dir):
        path = os.path.join(tmp_dir, "cache", "features")
        assert sorted(os.listdir(path)) == ["train.arrow"]

    def test_assemble_selects_and_attaches_labels(
        self, features, wav2vec2_processor, preprocessing_config
    ):
        texts = normalize_texts(features["transcription"], preprocessing_config)
        assignment = {"train": [0, 2], "test": [1]}
        labels = encode_labels(texts, wav2vec2_processor.tokenizer, [0, 1, 2])
        result = assemble_splits(
            features, assignment, labels, input_column="input_values", sampling_rate=16000
        )
        assert result.num_rows == {"train": 2, "test": 1}
        assert set(result["train"].column_names) == {
            "input_values", "input_length", "labels", "label_length", "example_index",
        }
        assert result["test"]["labels"][0] == wav2vec2_processor.tokenizer("world").input_ids
        assert result["train"]["example_index"] == [0, 2]
        assert result["test"]["example_index"] == [1]

    def test_encode_labels_skips_unused_rows(self, wav2vec2_processor):
        labels = encode_labels(["hello", "world"], wav2vec2_processor.tokenizer, [1])
        assert labels[0] == []
        assert labels[1] == wav2vec2_processor.tokenizer("world").input_ids

    def test_length_filters_apply_by_index(self, features, wav2vec2_processor):
        labels = encode_labels(
            ["hello", "world", "hello world"], wav2vec2_processor.tokenizer, [0, 1, 2]
        )
        too_short_audio = assemble_splits(
            features, {"train": [0, 1, 2]}, labels,
            input_column="input_values", sampling_rate=16000,
            max_audio_length_seconds=0.05,
        )
        assert too_short_audio.num_rows == {"train": 0}

        label_limit = assemble_splits(
            features, {"train": [0, 1, 2]}, labels,
            input_column="input_values", sampling_rate=16000,
            max_label_length=5,
        )
        assert label_limit.num_rows == {"train": 2}


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
            "label_length",
        }
        assert len(result["clean"]) == 1
        # input_length is measured on the raw audio for group_by_length batching;
        # for wav2vec2 that equals the feature length one-for-one
        example = result["clean"][0]
        assert example["input_length"] == len(example["input_values"])
        assert example["label_length"] == len(example["labels"])

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
