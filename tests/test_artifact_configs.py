"""Tests for src/artifact_configs.py."""

import os

import pytest
import yaml

from omegaconf import OmegaConf

from src.artifact_configs import (
    ArtifactConfig,
    DatasetConfig,
    ModelConfig,
    ProcessedDatasetConfig,
    _dict_diff,
    processed_cache_dirname,
    warn_on_legacy_processed_cache,
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


class TestProcessedCacheDirname:
    """Per-configuration processed subcache naming.

    The name must distinguish every configuration that produces different
    cached features or labels, so that caches for different models coexist
    under one dataset instead of invalidating each other.
    """

    @staticmethod
    def _args(model: dict, dataset: dict = None, sampling_rate: int = 16000):
        return OmegaConf.create(
            {
                "model": model,
                "audio": {"sampling_rate": sampling_rate},
                "dataset": dataset or {},
            }
        )

    def test_ctc_model_name(self):
        args = self._args(
            {"type": "wav2vec2", "short_name": "xlsr300m"},
            {"max_audio_length_seconds": 30.0},
        )
        assert processed_cache_dirname(args) == "processed_xlsr300m_sr16k_a30"

    def test_whisper_name_includes_label_limit_and_language(self):
        args = self._args(
            {
                "type": "whisper",
                "short_name": "whisper_small",
                "language": "af",
                "task": "transcribe",
            },
            {"max_audio_length_seconds": 30.0, "max_label_length": 448},
        )
        assert processed_cache_dirname(args) == (
            "processed_whisper-small_sr16k_a30_l448_af-transcribe"
        )

    def test_different_models_get_different_caches(self):
        dataset = {"max_audio_length_seconds": 30.0}
        ctc = processed_cache_dirname(
            self._args({"type": "wav2vec2", "short_name": "xlsr300m"}, dataset)
        )
        whisper = processed_cache_dirname(
            self._args(
                {"type": "whisper", "short_name": "whisper_small", "language": "af"},
                dataset,
            )
        )
        assert ctc != whisper

    def test_language_change_gets_a_different_cache(self):
        """Whisper's language token changes every cached label."""
        base = {"type": "whisper", "short_name": "whisper_small", "task": "transcribe"}
        afrikaans = processed_cache_dirname(self._args({**base, "language": "af"}))
        swahili = processed_cache_dirname(self._args({**base, "language": "sw"}))
        assert afrikaans != swahili

    def test_omits_unset_optional_fields(self):
        args = self._args({"type": "wav2vec2", "short_name": "xlsr300m"})
        assert processed_cache_dirname(args) == "processed_xlsr300m_sr16k"

    def test_falls_back_to_model_type_without_short_name(self):
        args = self._args({"type": "hubert"})
        assert processed_cache_dirname(args) == "processed_hubert_sr16k"

    def test_name_is_filesystem_safe(self):
        args = self._args({"type": "whisper", "short_name": "openai/whisper-small"})
        assert "/" not in processed_cache_dirname(args)


class TestLegacyProcessedCacheWarning:
    def test_warns_when_old_layout_present(self, tmp_path, capsys):
        os.makedirs(tmp_path / "processed")
        warn_on_legacy_processed_cache(str(tmp_path))
        assert "old layout" in capsys.readouterr().err

    def test_silent_when_absent(self, tmp_path, capsys):
        warn_on_legacy_processed_cache(str(tmp_path))
        assert capsys.readouterr().err == ""
