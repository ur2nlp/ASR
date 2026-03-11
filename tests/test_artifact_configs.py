"""Tests for src/artifact_configs.py."""

import os

import pytest
import yaml

from src.artifact_configs import (
    ArtifactConfig,
    DatasetConfig,
    ModelConfig,
    ProcessedDatasetConfig,
    _dict_diff,
)


class TestDictDiff:
    def test_identical_dicts(self):
        assert _dict_diff({"a": 1}, {"a": 1}) == []

    def test_value_difference(self):
        diffs = _dict_diff({"a": 1}, {"a": 2})
        assert len(diffs) == 1
        assert "a" in diffs[0]

    def test_missing_key(self):
        diffs = _dict_diff({"a": 1, "b": 2}, {"a": 1})
        assert len(diffs) == 1
        assert "cached" in diffs[0]

    def test_extra_key(self):
        diffs = _dict_diff({"a": 1}, {"a": 1, "b": 2})
        assert len(diffs) == 1
        assert "current" in diffs[0]

    def test_nested_diff(self):
        diffs = _dict_diff(
            {"a": {"b": 1}},
            {"a": {"b": 2}},
        )
        assert len(diffs) == 1
        assert "a.b" in diffs[0]

    def test_empty_dicts(self):
        assert _dict_diff({}, {}) == []


class TestArtifactConfigSaveAndCheck:
    def test_save_creates_file(self, tmp_dir):
        class TestConfig(ArtifactConfig):
            artifact_name = "Test"
            def to_dict(self):
                return {"key": "value"}

        config = TestConfig()
        path = os.path.join(tmp_dir, "config.yaml")
        config.save(path)
        assert os.path.exists(path)

        with open(path) as f:
            data = yaml.safe_load(f)
        assert data == {"key": "value"}

    def test_check_cached_matches(self, tmp_dir):
        class TestConfig(ArtifactConfig):
            artifact_name = "Test"
            def to_dict(self):
                return {"key": "value"}

        config = TestConfig()
        path = os.path.join(tmp_dir, "config.yaml")
        config.save(path)
        assert config.check_cached(path) is True

    def test_check_cached_no_file(self, tmp_dir):
        class TestConfig(ArtifactConfig):
            artifact_name = "Test"
            def to_dict(self):
                return {"key": "value"}

        config = TestConfig()
        path = os.path.join(tmp_dir, "nonexistent.yaml")
        assert config.check_cached(path) is True

    def test_check_cached_mismatch_raises(self, tmp_dir):
        class TestConfig(ArtifactConfig):
            artifact_name = "Test"
            def __init__(self, val):
                self.val = val
            def to_dict(self):
                return {"key": self.val}

        config1 = TestConfig("old")
        path = os.path.join(tmp_dir, "config.yaml")
        config1.save(path)

        config2 = TestConfig("new")
        with pytest.raises(ValueError, match="CONFIG MISMATCH"):
            config2.check_cached(path)

    def test_check_cached_mismatch_no_raise(self, tmp_dir):
        class TestConfig(ArtifactConfig):
            artifact_name = "Test"
            def __init__(self, val):
                self.val = val
            def to_dict(self):
                return {"key": self.val}

        config1 = TestConfig("old")
        path = os.path.join(tmp_dir, "config.yaml")
        config1.save(path)

        config2 = TestConfig("new")
        assert config2.check_cached(path, error_on_mismatch=False) is False


class TestDatasetConfig:
    def test_from_args(self, base_config):
        config = DatasetConfig.from_args(base_config)
        d = config.to_dict()
        assert d["type"] == "huggingface"
        assert d["id"] == "test_lang"

    def test_round_trip(self, tmp_dir, base_config):
        config = DatasetConfig.from_args(base_config)
        path = os.path.join(tmp_dir, "dataset.yaml")
        config.save(path)
        assert config.check_cached(path) is True


class TestProcessedDatasetConfig:
    def test_from_args(self, base_config):
        config = ProcessedDatasetConfig.from_args(base_config, vocab_size=32)
        d = config.to_dict()
        assert d["model_type"] == "wav2vec2"
        assert d["vocab_size"] == 32


class TestModelConfig:
    def test_from_args(self, base_config):
        config = ModelConfig.from_args(base_config, vocab_size=32)
        d = config.to_dict()
        assert d["model_type"] == "wav2vec2"
        assert "training_config" in d
