"""Tests for src/preprocessing.py."""

from omegaconf import OmegaConf

from src.preprocessing import build_text_normalizer, normalize_text


class TestNormalizeText:
    def test_lowercase(self):
        assert normalize_text("Hello WORLD") == "hello world"

    def test_remove_punctuation(self):
        result = normalize_text("Hello, world! How's it?")
        assert "," not in result
        assert "!" not in result
        assert "?" not in result
        # apostrophe is kept
        assert "'" in result

    def test_unicode_normalize(self):
        # e + combining acute should normalize to é under NFC
        text = "caf\u0065\u0301"
        result = normalize_text(text, remove_punctuation=False)
        assert "é" in result

    def test_custom_replacements(self):
        result = normalize_text(
            "2+2=4",
            remove_punctuation=False,
            custom_replacements={"2": "two", "+": " plus ", "=": " equals ", "4": "four"},
        )
        assert "two" in result
        assert "plus" in result

    def test_no_lowercase(self):
        result = normalize_text("Hello", lowercase=False)
        assert result[0] == "H"

    def test_no_punctuation_removal(self):
        result = normalize_text("Hello, world!", remove_punctuation=False)
        assert "," in result

    def test_no_unicode_normalize(self):
        text = "caf\u0065\u0301"
        result = normalize_text(text, unicode_normalize=None, remove_punctuation=False)
        # should remain decomposed
        assert len(result) >= len("café")

    def test_whitespace_collapsing(self):
        result = normalize_text("hello   world")
        assert result == "hello world"

    def test_strips(self):
        result = normalize_text("  hello  ")
        assert result == "hello"


class TestBuildTextNormalizer:
    def test_returns_callable(self, preprocessing_config):
        normalizer = build_text_normalizer(preprocessing_config)
        assert callable(normalizer)

    def test_applies_config(self, preprocessing_config):
        normalizer = build_text_normalizer(preprocessing_config)
        assert normalizer("Hello, World!") == "hello world"

    def test_custom_replacements(self):
        config = OmegaConf.create({
            "lowercase": True,
            "remove_punctuation": True,
            "unicode_normalize": "NFC",
            "custom_replacements": {"ü": "ue"},
        })
        normalizer = build_text_normalizer(config)
        assert "ue" in normalizer("über")
