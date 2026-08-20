"""Tests for src/processors.py."""

import pytest
from omegaconf import OmegaConf

from src.processors import (
    MODEL_SPECS,
    get_model_spec,
    resolve_language,
    setup_tokenizer,
)


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

    def test_whisper_spec(self):
        spec = get_model_spec("whisper")
        assert spec.input_column == "input_features"
        assert spec.objective == "seq2seq"

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown model type"):
            get_model_spec("nonexistent_model")

    def test_all_specs_have_required_fields(self):
        for name, spec in MODEL_SPECS.items():
            assert spec.model_class is not None, f"{name} missing model_class"
            assert spec.feature_extractor_class is not None
            assert spec.tokenizer_class is not None
            assert spec.processor_class is not None
            assert spec.collator_class is not None
            assert spec.trainer_class is not None
            assert spec.training_arguments_class is not None
            assert spec.input_column in ("input_values", "input_features")
            assert spec.objective in ("ctc", "seq2seq")
            assert spec.tokenizer_source in ("vocab", "pretrained")
            assert spec.feature_extractor_source in ("config", "pretrained")
            assert spec.feature_extractor_param_patterns, f"{name} has no fe patterns"
            assert "{index}" in spec.encoder_layer_param_pattern


class TestObjectiveConsistency:
    """The spec fields that must move together across an objective boundary."""

    def test_ctc_specs_are_internally_consistent(self):
        for name, spec in MODEL_SPECS.items():
            if spec.objective != "ctc":
                continue
            assert spec.tokenizer_source == "vocab", name
            assert not spec.uses_generation, name
            assert spec.supports_lm_decoding, name
            assert spec.learned_feature_extractor, name

    def test_seq2seq_specs_are_internally_consistent(self):
        for name, spec in MODEL_SPECS.items():
            if spec.objective != "seq2seq":
                continue
            # decoder embeddings are tied to the checkpoint's own vocabulary
            assert spec.tokenizer_source == "pretrained", name
            # WER must be computed over generated text, not teacher-forced logits
            assert spec.uses_generation, name
            # pyctcdecode shallow fusion has no autoregressive equivalent
            assert not spec.supports_lm_decoding, name

    def test_ctc_decode_kwargs_collapse_predictions_only(self):
        spec = get_model_spec("wav2vec2")
        assert spec.prediction_decode_kwargs == {"group_tokens": True}
        assert spec.label_decode_kwargs == {"group_tokens": False}

    def test_seq2seq_decode_kwargs_strip_special_tokens(self):
        spec = get_model_spec("whisper")
        assert spec.prediction_decode_kwargs == {"skip_special_tokens": True}
        assert spec.label_decode_kwargs == {"skip_special_tokens": True}


class TestSetupTokenizer:
    def test_vocab_source_requires_vocab_dir(self, base_config):
        with pytest.raises(ValueError, match="vocab_dir must be provided"):
            setup_tokenizer(base_config, train_texts=["hello"])

    def test_vocab_source_requires_texts_when_uncached(self, base_config, tmp_dir):
        with pytest.raises(ValueError, match="no train_texts"):
            setup_tokenizer(base_config, train_texts=None, vocab_dir=tmp_dir)

    def test_vocab_source_generates_and_caches(self, base_config, tmp_dir):
        tokenizer = setup_tokenizer(
            base_config,
            train_texts=["hello world"],
            vocab_dir=tmp_dir,
        )
        assert tokenizer.pad_token_id == 0
        # a second call reuses the cached vocab rather than regenerating
        reused = setup_tokenizer(base_config, train_texts=None, vocab_dir=tmp_dir)
        assert len(reused) == len(tokenizer)

    @pytest.mark.network
    def test_whisper_encodes_language_and_task_tokens(self, base_config):
        """Encoded text must actually carry the configured prefix tokens.

        Passing language/task to from_pretrained sets the attributes and makes
        `prefix_tokens` look correct, but does not rebuild the fast tokenizer's
        post-processor template -- so encoding silently drops the <|lang|> and
        <|task|> tokens and every cached label is wrong.
        """
        config = OmegaConf.merge(
            base_config,
            {
                "model": {
                    "type": "whisper",
                    "pretrained_name": "openai/whisper-tiny",
                    "language": "english",
                    "task": "transcribe",
                }
            },
        )
        tokenizer = setup_tokenizer(config)
        prefix = tokenizer.convert_ids_to_tokens(tokenizer("hello").input_ids[:4])

        assert prefix == [
            "<|startoftranscript|>",
            "<|en|>",
            "<|transcribe|>",
            "<|notimestamps|>",
        ]

    def test_unsupported_whisper_language_raises(self, base_config):
        config = OmegaConf.merge(
            base_config,
            {
                "model": {
                    "type": "whisper",
                    "pretrained_name": "openai/whisper-tiny",
                    "language": "zulu",
                    "task": "transcribe",
                }
            },
        )
        with pytest.raises(ValueError, match="no language token for"):
            setup_tokenizer(config)


class TestResolveLanguage:
    """`model.language` resolution for Whisper.

    None is rejected rather than passed through because the tokenizer and the
    decoder interpret it differently: the tokenizer drops the language token
    from the label prefix, while generate() detects and inserts one. `detect`
    is the explicit opt-in to that mismatch.
    """

    @staticmethod
    def _config(base_config, language):
        return OmegaConf.merge(
            base_config,
            {
                "model": {
                    "type": "whisper",
                    "pretrained_name": "openai/whisper-tiny",
                    "language": language,
                    "task": "transcribe",
                }
            },
        )

    def test_none_raises(self, base_config):
        spec = get_model_spec("whisper")
        with pytest.raises(ValueError, match="model.language is not set"):
            resolve_language(self._config(base_config, None), spec)

    def test_detect_returns_none_and_warns(self, base_config, capsys):
        spec = get_model_spec("whisper")
        assert resolve_language(self._config(base_config, "detect"), spec) is None
        assert "language-ID pass" in capsys.readouterr().err

    def test_supported_language_passes_through(self, base_config):
        spec = get_model_spec("whisper")
        assert resolve_language(self._config(base_config, "af"), spec) == "af"

    def test_ctc_models_are_unaffected_by_none(self, base_config):
        """CTC specs have no language notion, so None must stay legal there."""
        spec = get_model_spec("wav2vec2")
        assert resolve_language(self._config(base_config, None), spec) is None
