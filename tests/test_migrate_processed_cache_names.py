"""Tests for tools/migrate_processed_cache_names.py — which names get shortened."""

from tools.migrate_processed_cache_names import redundant_suffix


class TestRedundantSuffix:
    def test_drops_a_repeated_model_slug(self):
        name = "processed_whisper-medium-zu_sr16k_a30_l448_sw-transcribe_focus-v4k-whisper-medium-zu"
        new, slug = redundant_suffix(name)
        assert new == "processed_whisper-medium-zu_sr16k_a30_l448_sw-transcribe_focus-v4k"
        assert slug == "whisper-medium-zu"

    def test_leaves_a_focus_segment_that_has_no_model(self):
        """Caches predating per-model tokenizer keying already read focus-v4k."""
        assert redundant_suffix(
            "processed_whisper-small-zu_sr16k_a30_l448_sw-transcribe_focus-v4k"
        ) is None

    def test_leaves_a_cache_without_focus(self):
        assert redundant_suffix("processed_xlsr300m_sr16k_a30") is None

    def test_leaves_a_focus_segment_naming_a_different_model(self):
        """Only a slug matching this cache's own model is redundant."""
        assert redundant_suffix(
            "processed_whisper-small-zu_sr16k_focus-v4k-whisper-medium-zu"
        ) is None

    def test_ignores_directories_that_are_not_processed_caches(self):
        for name in ("untokenized", "focus", "vocab", "prepared", "processed"):
            assert redundant_suffix(name) is None

    def test_is_idempotent(self):
        name = "processed_whisper-medium-zu_sr16k_focus-v4k-whisper-medium-zu"
        once, _ = redundant_suffix(name)
        assert redundant_suffix(once) is None
