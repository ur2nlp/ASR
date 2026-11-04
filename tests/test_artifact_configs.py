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
    FeatureCacheConfig,
    ModelConfig,
    dict_diff,
    warn_on_legacy_processed_cache,
)
from src.processors import build_feature_extractor, get_model_spec


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

    The untokenized cache is written from the raw source *before* text
    normalization runs, so normalization cannot change it. Recording it there
    meant a preprocessing change hard-failed a cache it could not affect. The
    same holds for the feature cache now: labels are encoded per run from
    freshly normalized text and are not cached at all, so preprocessing must
    not key the features either -- a punctuation tweak must not force a
    re-extraction.
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

    def test_feature_record_ignores_preprocessing(self, base_config):
        other = base_config.copy()
        other.preprocessing.remove_punctuation = not base_config.preprocessing.remove_punctuation

        assert _feature_config(base_config).dirname() == _feature_config(other).dirname()


def _feature_config(args) -> FeatureCacheConfig:
    extractor = build_feature_extractor(args, get_model_spec(args.model.type))
    return FeatureCacheConfig.from_args(args, extractor)


class TestModelConfig:
    def test_from_args(self, base_config):
        config = ModelConfig.from_args(base_config, vocab_size=32)
        d = config.to_dict()
        assert d["model_type"] == "wav2vec2"
        assert "training_config" in d


class TestFeatureCacheConfig:
    """The shared feature cache is keyed on the audio and the extractor only.

    Everything else -- tokenizer, FOCUS vocabulary, language token, label and
    audio length limits, split assignment -- is applied per run by index, so it
    must *not* key the cache, or runs that could share the features would each
    extract their own copy.
    """

    def test_records_extractor_and_source(self, base_config):
        record = _feature_config(base_config).to_dict()
        assert record["input_column"] == "input_values"
        assert record["sampling_rate"] == 16000
        assert record["feature_extractor"]["do_normalize"] is True
        assert record["source"]["type"] == "huggingface"

    def test_drops_private_extractor_keys(self, base_config):
        record = _feature_config(base_config).to_dict()
        assert not any(key.startswith("_") for key in record["feature_extractor"])

    def test_name_names_the_extractor(self, base_config):
        name = _feature_config(base_config).dirname()
        assert name.startswith("features_wav2vec2featureextractor_")
        assert "/" not in name

    def test_shared_across_models_with_the_same_extractor(self, base_config):
        other = base_config.copy()
        other.model.short_name = "another-model"
        other.model.pretrained_name = "facebook/wav2vec2-base"
        assert _feature_config(base_config).dirname() == _feature_config(other).dirname()

    def test_ignores_length_limits(self, base_config):
        other = base_config.copy()
        other.dataset.max_audio_length_seconds = 10.0
        other.dataset.max_label_length = 448
        assert _feature_config(base_config).dirname() == _feature_config(other).dirname()

    def test_ignores_the_split_configuration(self, base_config):
        other = base_config.copy()
        other.split = {"enabled": True, "test": {"speaker": ["S01"]}}
        other.dataset.dev_size = 0.3
        assert _feature_config(base_config).dirname() == _feature_config(other).dirname()

    def test_extractor_settings_change_the_cache(self, base_config):
        other = base_config.copy()
        other.audio.do_normalize = False
        assert _feature_config(base_config).dirname() != _feature_config(other).dirname()

    def test_source_changes_the_cache(self, base_config):
        other = base_config.copy()
        other.dataset.name = "another_dataset"
        assert _feature_config(base_config).dirname() != _feature_config(other).dirname()

    def test_source_splits_do_not_change_the_name(self, base_config):
        """They are only known after reading the source, after the name is needed."""
        config = _feature_config(base_config)
        before = config.dirname()
        config.source_splits = ["train", "test"]
        assert config.dirname() == before


class TestLegacyProcessedCacheWarning:
    def test_warns_when_old_layout_present(self, tmp_path, capsys):
        os.makedirs(tmp_path / "processed")
        warn_on_legacy_processed_cache(str(tmp_path))
        assert "old layout" in capsys.readouterr().err

    def test_warns_on_per_model_processed_subcaches(self, tmp_path, capsys):
        os.makedirs(tmp_path / "processed_whisper-small_sr16k_a30")
        os.makedirs(tmp_path / "features_whisperfeatureextractor_0123abcd")
        warn_on_legacy_processed_cache(str(tmp_path))
        err = capsys.readouterr().err
        assert "processed_whisper-small_sr16k_a30" in err
        assert "features_" not in err.split("Features now")[0]

    def test_silent_when_absent(self, tmp_path, capsys):
        warn_on_legacy_processed_cache(str(tmp_path))
        assert capsys.readouterr().err == ""
