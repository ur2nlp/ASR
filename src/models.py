"""Model setup: loading, dropout configuration, freezing, and vocab resize.

Loads a pretrained CTC model, overrides dropout/masking parameters, sets the
vocabulary size from the tokenizer, and optionally freezes the feature
extractor and/or encoder layers.
"""

import sys

from omegaconf import DictConfig
from transformers import ProcessorMixin

from src.processors import get_model_spec


def setup_model(args: DictConfig, processor: ProcessorMixin):
    """Load and configure a pretrained model for CTC fine-tuning.

    Args:
        args: Full Hydra config (needs args.model.*).
        processor: The combined processor (to determine vocab_size and
            pad_token_id).

    Returns:
        A configured HuggingFace model ready for training.
    """
    spec = get_model_spec(args.model.type)
    model_class = spec.model_class

    vocab_size = len(processor.tokenizer)
    pad_token_id = processor.tokenizer.pad_token_id

    config_overrides = {
        "attention_dropout": args.training.attention_dropout,
        "hidden_dropout": args.training.hidden_dropout,
        "feat_proj_dropout": args.training.feat_proj_dropout,
        "mask_time_prob": args.training.mask_time_prob,
        "mask_feature_prob": args.training.get("mask_feature_prob", 0.0),
        "layerdrop": args.training.layerdrop,
        "ctc_loss_reduction": args.training.ctc_loss_reduction,
        # zero out non-finite CTC losses (transcript longer than the audio can
        # emit); important for auto-segmented data with imprecise boundaries
        "ctc_zero_infinity": args.training.get("ctc_zero_infinity", False),
        "pad_token_id": pad_token_id,
        "vocab_size": vocab_size,
    }

    # ignore_mismatched_sizes lets us adapt from an already-fine-tuned CTC
    # checkpoint whose lm_head size differs from our new vocab: the mismatched
    # head is freshly initialized while the rest of the weights load normally.
    # Base (non-CTC) checkpoints have no lm_head, so this is a no-op for them.
    model = model_class.from_pretrained(
        args.model.pretrained_name,
        ignore_mismatched_sizes=True,
        **config_overrides,
    )

    # gradient checkpointing requires disabling cache
    if args.training.get("gradient_checkpointing", False):
        model.config.use_cache = False

    # freeze feature extractor
    if args.training.get("freeze_feature_extractor", True):
        _freeze_feature_extractor(model)

    # freeze first N encoder layers
    freeze_layers = args.training.get("freeze_encoder_layers", 0)
    if freeze_layers > 0:
        _freeze_encoder_layers(model, freeze_layers)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"Model loaded: {args.model.pretrained_name} "
        f"(vocab_size={vocab_size}, "
        f"trainable={trainable:,}/{total:,} params)",
        file=sys.stderr,
    )

    return model


def _freeze_feature_extractor(model) -> None:
    """Freeze the CNN feature extractor weights."""
    # works for wav2vec2, hubert, and w2v-bert via parameter name matching
    frozen_count = 0
    for name, param in model.named_parameters():
        if "feature_extractor" in name or "feature_projection" in name:
            param.requires_grad = False
            frozen_count += 1
    print(f"Froze {frozen_count} feature extractor parameters", file=sys.stderr)


def _freeze_encoder_layers(model, num_layers: int) -> None:
    """Freeze the first `num_layers` transformer encoder layers."""
    frozen_count = 0
    for name, param in model.named_parameters():
        for layer_idx in range(num_layers):
            if f"layers.{layer_idx}." in name or f"encoder.layers.{layer_idx}." in name:
                param.requires_grad = False
                frozen_count += 1
                break
    print(f"Froze first {num_layers} encoder layers ({frozen_count} params)", file=sys.stderr)


def unfreeze_all(model) -> None:
    """Unfreeze all model parameters (used by unfreeze callbacks)."""
    for param in model.parameters():
        param.requires_grad = True
    print("Unfroze all model parameters", file=sys.stderr)
