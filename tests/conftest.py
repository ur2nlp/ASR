"""Shared test fixtures for ASR tests."""

import json
import os
import tempfile

import pytest
from omegaconf import OmegaConf


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_texts():
    """Sample normalized transcription texts for vocab tests."""
    return [
        "hello world",
        "this is a test",
        "hello again",
    ]


@pytest.fixture
def sample_vocab():
    """A small character vocab dictionary."""
    return {
        "[PAD]": 0,
        "[UNK]": 1,
        "|": 2,
        "a": 3,
        "d": 4,
        "e": 5,
        "g": 6,
        "h": 7,
        "i": 8,
        "l": 9,
        "n": 10,
        "o": 11,
        "r": 12,
        "s": 13,
        "t": 14,
        "w": 15,
    }


@pytest.fixture
def vocab_path(tmp_dir, sample_vocab):
    """Write sample vocab to a temp file and return its path."""
    path = os.path.join(tmp_dir, "vocab.json")
    with open(path, "w") as f:
        json.dump(sample_vocab, f)
    return path


@pytest.fixture
def preprocessing_config():
    """Default preprocessing config as DictConfig."""
    return OmegaConf.create({
        "lowercase": True,
        "remove_punctuation": True,
        "unicode_normalize": "NFC",
        "custom_replacements": {},
    })


@pytest.fixture
def base_config():
    """Minimal base Hydra config for testing."""
    return OmegaConf.create({
        "seed": 1,
        "model": {
            "type": "wav2vec2",
            "pretrained_name": "facebook/wav2vec2-xls-r-300m",
            "short_name": "xlsr300m",
        },
        "audio": {
            "sampling_rate": 16000,
            "feature_size": 1,
            "do_normalize": True,
            "return_attention_mask": True,
        },
        "dataset": {
            "type": "huggingface",
            "id": "test_lang",
            "name": "test_dataset",
            "config": None,
            "cache_dir": "data/test",
            "audio_column": "audio",
            "text_column": "transcription",
            "dev_size": 0.1,
            "max_audio_length_seconds": 30.0,
        },
        "preprocessing": {
            "lowercase": True,
            "remove_punctuation": True,
            "unicode_normalize": "NFC",
            "custom_replacements": {},
        },
        "training": {
            "name": "test",
            "num_train_epochs": 1,
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "learning_rate": 3e-5,
            "lr_scheduler_type": "linear",
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "save_total_limit": 1,
            "load_best_model_at_end": True,
            "metric_for_best_model": "wer",
            "greater_is_better": False,
            "logging_steps": 10,
            "bf16": False,
            "gradient_checkpointing": False,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "hidden_dropout": 0.1,
            "feat_proj_dropout": 0.1,
            "mask_time_prob": 0.05,
            "layerdrop": 0.0,
            "ctc_loss_reduction": "mean",
            "freeze_feature_extractor": True,
            "freeze_encoder_layers": 0,
            "max_steps": 100,
            "early_stopping_patience": 3,
            "early_stopping_delay_ratio": 0.0,
            "unfreeze_step_ratio": None,
        },
        "lm": {
            "enabled": False,
            "arpa_path": None,
        },
        "output_dir": "models",
    })
