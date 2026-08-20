"""Tests for src/models.py.

These cover the config-override construction and the freezing helpers, which
are where the CTC / seq2seq split actually bites. Loading a real pretrained
checkpoint is left to integration testing so the unit suite stays network-free.
"""

import pytest
import torch
from omegaconf import OmegaConf
from transformers import WhisperConfig, Wav2Vec2Config

from src.models import (
    _build_config_overrides,
    _freeze_encoder,
    _freeze_encoder_layers,
    _freeze_feature_extractor,
)
from src.processors import get_model_spec
from src.vocab import create_ctc_tokenizer


@pytest.fixture
def processor(base_config, vocab_path):
    from src.processors import setup_processor

    return setup_processor(base_config, create_ctc_tokenizer(vocab_path))


@pytest.fixture
def whisper_config(base_config):
    return OmegaConf.merge(
        base_config,
        {
            "model": {
                "type": "whisper",
                "pretrained_name": "openai/whisper-tiny",
                "language": None,
                "task": "transcribe",
            },
            "training": {
                "activation_dropout": 0.05,
                "encoder_layerdrop": 0.1,
                "decoder_layerdrop": 0.2,
            },
        },
    )


class TestConfigOverrides:
    def test_ctc_overrides_set_vocab_size(self, base_config, processor):
        spec = get_model_spec("wav2vec2")
        overrides = _build_config_overrides(base_config, processor, spec)

        assert overrides["vocab_size"] == len(processor.tokenizer)
        assert overrides["ctc_loss_reduction"] == "mean"

    def test_ctc_overrides_are_all_valid_config_fields(self, base_config, processor):
        spec = get_model_spec("wav2vec2")
        overrides = _build_config_overrides(base_config, processor, spec)

        valid_fields = Wav2Vec2Config().to_dict()
        for key in overrides:
            assert key in valid_fields, f"{key} is not a Wav2Vec2Config field"

    def test_seq2seq_overrides_leave_vocab_alone(self, whisper_config, processor):
        spec = get_model_spec("whisper")
        overrides = _build_config_overrides(whisper_config, processor, spec)

        # resizing the decoder embeddings would discard the pretrained decoder
        assert "vocab_size" not in overrides
        assert "pad_token_id" not in overrides

    def test_seq2seq_overrides_omit_ctc_only_keys(self, whisper_config, processor):
        spec = get_model_spec("whisper")
        overrides = _build_config_overrides(whisper_config, processor, spec)

        for ctc_only_key in (
            "ctc_loss_reduction",
            "ctc_zero_infinity",
            "feat_proj_dropout",
            "hidden_dropout",
            "layerdrop",
        ):
            assert ctc_only_key not in overrides

    def test_seq2seq_overrides_are_all_valid_config_fields(
        self, whisper_config, processor
    ):
        spec = get_model_spec("whisper")
        overrides = _build_config_overrides(whisper_config, processor, spec)

        valid_fields = WhisperConfig().to_dict()
        for key in overrides:
            assert key in valid_fields, f"{key} is not a WhisperConfig field"

    def test_seq2seq_splits_encoder_and_decoder_layerdrop(
        self, whisper_config, processor
    ):
        spec = get_model_spec("whisper")
        overrides = _build_config_overrides(whisper_config, processor, spec)

        assert overrides["encoder_layerdrop"] == 0.1
        assert overrides["decoder_layerdrop"] == 0.2

    def test_seq2seq_layerdrop_falls_back_to_shared_value(self, base_config, processor):
        config = OmegaConf.merge(
            base_config,
            {"model": {"type": "whisper", "pretrained_name": "openai/whisper-tiny"}},
        )
        config.training.layerdrop = 0.15
        spec = get_model_spec("whisper")
        overrides = _build_config_overrides(config, processor, spec)

        assert overrides["encoder_layerdrop"] == 0.15
        assert overrides["decoder_layerdrop"] == 0.15

    def test_spec_augment_enabled_only_when_masking_requested(
        self, whisper_config, processor
    ):
        spec = get_model_spec("whisper")

        whisper_config.training.mask_time_prob = 0.0
        whisper_config.training.mask_feature_prob = 0.0
        assert not _build_config_overrides(whisper_config, processor, spec)[
            "apply_spec_augment"
        ]

        whisper_config.training.mask_time_prob = 0.05
        assert _build_config_overrides(whisper_config, processor, spec)[
            "apply_spec_augment"
        ]


class _StubModel(torch.nn.Module):
    """A module whose parameter names mimic an encoder-decoder ASR model."""

    def __init__(self):
        super().__init__()
        self.model = torch.nn.ModuleDict({
            "encoder": torch.nn.ModuleDict({
                "conv1": torch.nn.Linear(2, 2),
                "conv2": torch.nn.Linear(2, 2),
                "layers": torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(3)]),
            }),
            "decoder": torch.nn.ModuleDict({
                "layers": torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(3)]),
            }),
        })


def _trainable(model) -> set[str]:
    return {name for name, param in model.named_parameters() if param.requires_grad}


class TestFreezing:
    def test_freeze_feature_extractor_uses_spec_patterns(self):
        model = _StubModel()
        _freeze_feature_extractor(model, get_model_spec("whisper"))

        trainable = _trainable(model)
        assert not any("encoder.conv" in name for name in trainable)
        assert any("encoder.layers" in name for name in trainable)

    def test_freeze_encoder_layers_spares_the_decoder(self):
        model = _StubModel()
        _freeze_encoder_layers(model, get_model_spec("whisper"), num_layers=2)

        trainable = _trainable(model)
        # first two encoder layers frozen, third still trainable
        assert "model.encoder.layers.0.weight" not in trainable
        assert "model.encoder.layers.1.weight" not in trainable
        assert "model.encoder.layers.2.weight" in trainable
        # decoder layers share the bare `layers.N.` naming and must be untouched
        assert "model.decoder.layers.0.weight" in trainable
        assert "model.decoder.layers.1.weight" in trainable

    def test_freeze_encoder_freezes_whole_stack(self):
        model = _StubModel()
        _freeze_encoder(model)

        trainable = _trainable(model)
        assert not any("encoder." in name for name in trainable)
        assert any("decoder." in name for name in trainable)
