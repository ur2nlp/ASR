"""Tests for tools/migrate_model_short_name.py — matching and renaming.

The matching is the part worth guarding: this tool renames cache directories,
so a pattern that is too greedy moves a cache that belongs to a different model
and the run that owns it silently rebuilds.
"""

import os

import pytest
import yaml

from tools.migrate_model_short_name import (
    focus_dirs,
    processed_dirs,
    record_needs_rewrite,
    renamed,
    rewrite_record,
)

OLD = "whisper-medium-zu"
NEW = "whisper-medium"


@pytest.fixture
def cache_tree(tmp_path):
    """A dataset cache holding one processed subcache and one FOCUS tokenizer."""
    dataset = tmp_path / "zulu"
    (dataset / f"processed_{OLD}_sr16k_a30_l448").mkdir(parents=True)
    (dataset / "focus" / f"focus-v4k-unigram-{OLD}").mkdir(parents=True)
    (dataset / "untokenized").mkdir()
    return str(dataset)


class TestMatching:
    def test_finds_the_processed_subcache(self, cache_tree):
        found = [os.path.basename(p) for p in processed_dirs(cache_tree, OLD)]
        assert found == [f"processed_{OLD}_sr16k_a30_l448"]

    def test_finds_the_focus_tokenizer(self, cache_tree):
        found = [os.path.basename(p) for p in focus_dirs(cache_tree, OLD)]
        assert found == [f"focus-v4k-unigram-{OLD}"]

    def test_a_longer_model_name_is_not_a_match(self, cache_tree):
        """`whisper-medium-zu` must not claim `whisper-medium-zu-v2`."""
        os.mkdir(os.path.join(cache_tree, f"processed_{OLD}-v2_sr16k"))
        found = [os.path.basename(p) for p in processed_dirs(cache_tree, OLD)]
        assert found == [f"processed_{OLD}_sr16k_a30_l448"]

    def test_a_shorter_model_name_is_not_a_match(self, cache_tree):
        """`whisper-medium` must not claim `whisper-medium-zu`'s caches."""
        assert processed_dirs(cache_tree, NEW) == []
        assert focus_dirs(cache_tree, NEW) == []

    def test_a_processed_subcache_with_no_suffix_still_matches(self, tmp_path):
        dataset = tmp_path / "zulu"
        (dataset / f"processed_{OLD}").mkdir(parents=True)
        found = [os.path.basename(p) for p in processed_dirs(str(dataset), OLD)]
        assert found == [f"processed_{OLD}"]

    def test_a_dataset_without_focus_is_fine(self, tmp_path):
        dataset = tmp_path / "zulu"
        dataset.mkdir()
        assert focus_dirs(str(dataset), OLD) == []


class TestRenaming:
    def test_processed_keeps_its_suffix(self):
        new = renamed(f"/data/zulu/processed_{OLD}_sr16k_a30", OLD, NEW)
        assert os.path.basename(new) == f"processed_{NEW}_sr16k_a30"

    def test_focus_keeps_its_vocabulary_prefix(self):
        new = renamed(f"/data/zulu/focus/focus-v4k-unigram-{OLD}", OLD, NEW)
        assert os.path.basename(new) == "focus-v4k-unigram-whisper-medium"

    def test_renaming_is_reversible(self):
        original = f"/data/zulu/processed_{OLD}_sr16k"
        assert renamed(renamed(original, OLD, NEW), NEW, OLD) == original


class TestRecordRewrite:
    def _write(self, directory, value):
        path = os.path.join(directory, "config.yaml")
        with open(path, "w") as config_file:
            yaml.dump({"pretrained_name": "openai/whisper-medium",
                       "focus_tokenizer_id": value}, config_file)
        return path

    def test_detects_a_stale_tokenizer_id(self, tmp_path):
        path = self._write(str(tmp_path), f"focus-v4k-unigram-{OLD}")
        assert record_needs_rewrite(path, OLD) == f"focus-v4k-unigram-{OLD}"

    def test_ignores_a_record_naming_another_model(self, tmp_path):
        path = self._write(str(tmp_path), "focus-v4k-unigram-xlsr300m")
        assert record_needs_rewrite(path, OLD) is None

    def test_ignores_a_record_without_focus(self, tmp_path):
        path = self._write(str(tmp_path), None)
        assert record_needs_rewrite(path, OLD) is None

    def test_ignores_a_missing_record(self, tmp_path):
        assert record_needs_rewrite(str(tmp_path / "absent.yaml"), OLD) is None

    def test_rewrite_swaps_only_the_model_slug(self, tmp_path):
        path = self._write(str(tmp_path), f"focus-v4k-unigram-{OLD}")
        rewrite_record(path, OLD, NEW)
        with open(path) as config_file:
            data = yaml.safe_load(config_file)
        assert data["focus_tokenizer_id"] == "focus-v4k-unigram-whisper-medium"
        assert data["pretrained_name"] == "openai/whisper-medium"
