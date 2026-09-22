"""Tests for src/artifact_configs.py.

`dict_diff` and `ArtifactConfig` now come from `lapt-core` and are imported
through `src.artifact_configs`, which re-exports them. These tests are kept
pointed at that re-export deliberately: they cover the behaviour ASR depends
on, from the angle ASR consumes it, and would still catch a bad upgrade of the
shared package.
"""

import os

import pytest
import yaml

from omegaconf import OmegaConf

from src.sources import source_config_record
from src.artifact_configs import (
    ArtifactConfig,
    ModelConfig,
    ProcessedDatasetConfig,
    dict_diff,
    processed_cache_dirname,
    warn_on_legacy_processed_cache,
)


class TestDictDiff:
    def test_identical_dicts(self):
        assert dict_diff({"a": 1}, {"a": 1}) == []

    def test_value_difference(self):
        diffs = dict_diff({"a": 1}, {"a": 2})
        assert len(diffs) == 1
        assert "a" in diffs[0]

    def test_missing_key(self):
        diffs = dict_diff({"a": 1, "b": 2}, {"a": 1})
        assert len(diffs) == 1
        assert "cached" in diffs[0]

    def test_extra_key(self):
        diffs = dict_diff({"a": 1}, {"a": 1, "b": 2})
        assert len(diffs) == 1
        assert "current" in diffs[0]

    def test_nested_diff(self):
        diffs = dict_diff(
            {"a": {"b": 1}},
            {"a": {"b": 2}},
        )
        assert len(diffs) == 1
        assert "a.b" in diffs[0]

    def test_empty_dicts(self):
        assert dict_diff({}, {}) == []


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


class TestUntokenizedRecord:
    """What the untokenized stage records, now that the source artifact owns it.

    `DatasetConfig` is gone: each source in `src/sources/` declares its own
    `config()`, which is both the cache key and the saved record.
    """

    def test_names_the_type_and_the_source(self, base_config):
        record = source_config_record(base_config.dataset, base_config.seed)
        assert record["type"] == "huggingface"
        assert record["name"] == "test_dataset"

    def test_round_trip(self, tmp_dir, base_config):
        record = source_config_record(base_config.dataset, base_config.seed)
        path = os.path.join(tmp_dir, "config.yaml")
        with open(path, "w") as config_file:
            yaml.dump(record, config_file)
        with open(path) as config_file:
            assert yaml.safe_load(config_file) == record


class TestPairedDoesNotKeyOnColumnNames:
    """Regression guard for the mirror image of the preprocessing bug.

    Every other source keys its cache on `audio_column`/`text_column`, and
    rightly so: those name the columns the data arrives with, and `_standardize`
    renames them before the cache is written. A paired corpus has no incoming
    column names -- `build_paired_split` invents `audio`/`transcription` from
    file stems -- so recording them keyed the cache on something that could not
    change its contents. Harmless in direction (it forces a rebuild rather than
    reusing stale data) but it made a no-op setting look load-bearing.
    """

    def _paired_config(self, **overrides):
        config = {"type": "paired", "path": "/data/corpus"}
        config.update(overrides)
        return config

    def test_record_omits_the_column_names(self):
        record = source_config_record(self._paired_config())
        assert "audio_column" not in record
        assert "text_column" not in record

    def test_record_still_keys_what_does_matter(self):
        record = source_config_record(self._paired_config())
        assert record["type"] == "paired"
        assert record["path"] == "/data/corpus"
        assert record["audio_ext"] == ".wav"
        assert record["transcript_ext"] == ".txt"
        assert record["recursive"] is False

    def test_changing_a_column_name_does_not_invalidate_the_cache(self):
        baseline = source_config_record(self._paired_config())
        renamed = source_config_record(
            self._paired_config(audio_column="wav", text_column="text")
        )
        assert baseline == renamed

    def test_changing_an_extension_does_invalidate_the_cache(self):
        baseline = source_config_record(self._paired_config())
        other = source_config_record(self._paired_config(audio_ext=".flac"))
        assert baseline != other

    def test_a_non_default_column_name_warns(self, capsys):
        source_config_record(self._paired_config(audio_column="wav"))
        assert "has no effect on a 'paired' source" in capsys.readouterr().err

    def test_inherited_defaults_do_not_warn(self):
        import io as _io
        import contextlib
        stderr = _io.StringIO()
        with contextlib.redirect_stderr(stderr):
            source_config_record(
                self._paired_config(audio_column="audio", text_column="transcription")
            )
        assert stderr.getvalue() == ""


class TestPreprocessingIsRecordedOnTheStageItAffects:
    """Regression guard for a mis-stated dependency.

    The untokenized cache is written from the raw source *before*
    `normalize_dataset` runs, so text normalization cannot change it. The
    processed cache holds label-encoded transcripts, so normalization changes
    it completely. Recording it on the untokenized stage meant a preprocessing
    change hard-failed a cache it could not affect, while the cache it did
    affect was reused silently -- `ProcessedDatasetConfig` did not track it and
    `processed_cache_dirname` did not either, so nothing noticed.
    """

    def test_untokenized_record_ignores_preprocessing(self, base_config):
        other = base_config.copy()
        other.preprocessing.remove_punctuation = not base_config.preprocessing.remove_punctuation

        assert source_config_record(base_config.dataset, base_config.seed) == \
            source_config_record(other.dataset, other.seed)

    def test_untokenized_record_ignores_seed(self, base_config):
        other = base_config.copy()
        other.seed = base_config.seed + 1

        assert source_config_record(base_config.dataset, base_config.seed) == \
            source_config_record(other.dataset, other.seed)

    def test_processed_record_tracks_preprocessing(self, base_config):
        other = base_config.copy()
        other.preprocessing.remove_punctuation = not base_config.preprocessing.remove_punctuation

        assert ProcessedDatasetConfig.from_args(base_config, 32).to_dict() != \
            ProcessedDatasetConfig.from_args(other, 32).to_dict()


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
