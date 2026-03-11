"""Tests for src/processors.py."""

import pytest

from src.processors import MODEL_SPECS, get_model_spec


class TestModelSpecs:
    def test_wav2vec2_spec(self):
        spec = get_model_spec("wav2vec2")
        assert spec.input_column == "input_values"

    def test_hubert_spec(self):
        spec = get_model_spec("hubert")
        assert spec.input_column == "input_values"

    def test_w2v_bert_spec(self):
        spec = get_model_spec("w2v_bert")
        assert spec.input_column == "input_features"

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown model type"):
            get_model_spec("nonexistent_model")

    def test_all_specs_have_required_fields(self):
        for name, spec in MODEL_SPECS.items():
            assert spec.model_class is not None, f"{name} missing model_class"
            assert spec.feature_extractor_class is not None
            assert spec.processor_class is not None
            assert spec.input_column in ("input_values", "input_features")
