"""Model setup: loading, dropout configuration, freezing, and vocab resize.

Loads a pretrained model, applies the config overrides appropriate to its
training objective, and optionally freezes the audio front end, whole encoder,
or the first N encoder layers.

Two objectives are supported and differ in one important way. A CTC model's
output head is sized to a character vocabulary generated from the training
data, so `vocab_size` is overridden and a mismatched head is re-initialized.
A seq2seq model's decoder embeddings are tied to the checkpoint's own subword
vocabulary, so its size must be left alone.
"""

import sys

from omegaconf import DictConfig
from transformers import ProcessorMixin

from src import focus
from src.processors import ModelSpec, get_model_spec, resolve_language


def setup_model(
    args: DictConfig,
    processor: ProcessorMixin,
    cache_dir: str | None = None,
):
    """Load and configure a pretrained model for fine-tuning.

    Args:
        args: Full Hydra config (needs args.model.*).
        processor: The combined processor (to determine vocab_size and
            pad_token_id).
        cache_dir: The dataset cache directory. Required only when
            `focus.enabled` is set, which loads its cached tokenizer and
            embeddings from under it.

    Returns:
        A configured HuggingFace model ready for training.
    """
    spec = get_model_spec(args.model.type)

    config_overrides = _build_config_overrides(args, processor, spec)

    # ignore_mismatched_sizes lets us adapt from an already-fine-tuned CTC
    # checkpoint whose lm_head size differs from our new vocab: the mismatched
    # head is freshly initialized while the rest of the weights load normally.
    # Base (non-CTC) checkpoints have no lm_head, so this is a no-op for them.
    model = spec.model_class.from_pretrained(
        args.model.pretrained_name,
        ignore_mismatched_sizes=True,
        **config_overrides,
    )

    # FOCUS must run before generation is configured: it rewrites the token ids
    # that `_configure_generation` then reads and pins.
    if focus.is_enabled(args):
        if cache_dir is None:
            raise ValueError(
                "focus.enabled=true requires cache_dir so setup_model can find "
                "the FOCUS tokenizer and cached embeddings."
            )
        focus.apply_to_model(args, model, processor.tokenizer, spec, cache_dir)

    if spec.uses_generation:
        _configure_generation(args, model, spec)

    # gradient checkpointing requires disabling cache
    if args.training.get("gradient_checkpointing", False):
        model.config.use_cache = False

    # freeze the audio front end; models whose front end is deterministic
    # (Whisper's mel spectrogram) default to leaving its conv stack trainable
    freeze_front_end = args.training.get(
        "freeze_feature_extractor", spec.learned_feature_extractor
    )
    if freeze_front_end:
        _freeze_feature_extractor(model, spec)

    # freeze the whole encoder
    if args.training.get("freeze_encoder", False):
        _freeze_encoder(model)

    # freeze first N encoder layers
    freeze_layers = args.training.get("freeze_encoder_layers", 0)
    if freeze_layers > 0:
        _freeze_encoder_layers(model, spec, freeze_layers)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"Model loaded: {args.model.pretrained_name} "
        f"(objective={spec.objective}, "
        f"vocab_size={model.config.vocab_size}, "
        f"trainable={trainable:,}/{total:,} params)",
        file=sys.stderr,
    )

    return model


# ---------------------------------------------------------------------------
# Objective-specific config overrides
# ---------------------------------------------------------------------------

def _build_config_overrides(
    args: DictConfig,
    processor: ProcessorMixin,
    spec: ModelSpec,
) -> dict:
    """Select and build the model config overrides for this objective."""
    builders = {
        "ctc": _ctc_config_overrides,
        "seq2seq": _seq2seq_config_overrides,
    }
    if spec.objective not in builders:
        raise ValueError(
            f"Unknown objective '{spec.objective}' for model type "
            f"'{args.model.type}'. Available: {list(builders.keys())}"
        )
    return builders[spec.objective](args, processor)


def _ctc_config_overrides(args: DictConfig, processor: ProcessorMixin) -> dict:
    """Config overrides for CTC models (wav2vec2, HuBERT, W2V-BERT)."""
    return {
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
        "pad_token_id": processor.tokenizer.pad_token_id,
        "vocab_size": len(processor.tokenizer),
    }


def _seq2seq_config_overrides(args: DictConfig, processor: ProcessorMixin) -> dict:
    """Config overrides for encoder-decoder models (Whisper).

    Deliberately omits `vocab_size` and `pad_token_id`: the decoder's input and
    output embeddings are tied to the pretrained subword vocabulary, and
    resizing them would discard the pretrained decoder. The CTC-only knobs
    (`ctc_loss_reduction`, `feat_proj_dropout`, `layerdrop`) have no
    counterpart here; the encoder and decoder carry separate layerdrop rates,
    both defaulting to the CTC `layerdrop` value for config compatibility.
    """
    layerdrop = args.training.get("layerdrop", 0.0)
    return {
        "dropout": args.training.get("dropout", 0.0),
        "attention_dropout": args.training.attention_dropout,
        "activation_dropout": args.training.get("activation_dropout", 0.0),
        "encoder_layerdrop": args.training.get("encoder_layerdrop", layerdrop),
        "decoder_layerdrop": args.training.get("decoder_layerdrop", layerdrop),
        # SpecAugment over the encoder's input features; off by default in the
        # pretrained config, and a useful regularizer on small corpora
        "mask_time_prob": args.training.mask_time_prob,
        "mask_feature_prob": args.training.get("mask_feature_prob", 0.0),
        "apply_spec_augment": (
            args.training.mask_time_prob > 0.0
            or args.training.get("mask_feature_prob", 0.0) > 0.0
        ),
    }


def _configure_generation(args: DictConfig, model, spec: ModelSpec) -> None:
    """Pin the generation-time language and task for a multilingual decoder.

    Whisper decides what language to transcribe from special tokens prefixed to
    the decoder input. During fine-tuning on a single language we want those
    tokens fixed rather than predicted, both so evaluation decodes in the right
    language and so generation does not waste beams on language ID.

    The language is resolved through `processors.resolve_language` so that the
    decoder prefix matches the one the tokenizer wrote into the training labels.
    A None here means the user opted into language detection explicitly, and has
    already been warned.

    `forced_decoder_ids` is the pre-4.34 mechanism for this and is now
    superseded by `generation_config.language`/`task`; leaving a stale value in
    place would conflict with them, so it is cleared.
    """
    language = resolve_language(args, spec)
    task = args.model.get("task", "transcribe")

    if language is not None:
        model.generation_config.language = language
    model.generation_config.task = task
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None

    # The pretrained suppress lists block tokens that are unlikely in the
    # pretraining languages, which can be wrong for an adapted language.
    # Setting model.suppress_tokens to [] in the config clears them.
    suppress_tokens = args.model.get("suppress_tokens", None)
    if suppress_tokens is not None:
        model.generation_config.suppress_tokens = list(suppress_tokens)
        model.config.suppress_tokens = list(suppress_tokens)

    print(
        f"Generation configured: language={language}, task={task}, "
        f"suppress_tokens={'default' if suppress_tokens is None else suppress_tokens}",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Freezing
# ---------------------------------------------------------------------------

def _freeze_feature_extractor(model, spec: ModelSpec) -> None:
    """Freeze the audio front end, matching on spec-declared param patterns."""
    frozen_count = 0
    for name, param in model.named_parameters():
        if any(pattern in name for pattern in spec.feature_extractor_param_patterns):
            param.requires_grad = False
            frozen_count += 1
    print(f"Froze {frozen_count} feature extractor parameters", file=sys.stderr)


def _freeze_encoder(model) -> None:
    """Freeze the entire encoder stack, leaving the decoder/head trainable."""
    frozen_count = 0
    for name, param in model.named_parameters():
        if "encoder." in name:
            param.requires_grad = False
            frozen_count += 1
    print(f"Froze entire encoder ({frozen_count} params)", file=sys.stderr)


def _freeze_encoder_layers(model, spec: ModelSpec, num_layers: int) -> None:
    """Freeze the first `num_layers` transformer encoder layers.

    The match is anchored on the encoder (via the spec's
    `encoder_layer_param_pattern`) so that an encoder-decoder model's decoder
    layers, which share the bare `layers.{index}.` naming, are left alone.
    """
    patterns = [
        spec.encoder_layer_param_pattern.format(index=layer_index)
        for layer_index in range(num_layers)
    ]

    frozen_count = 0
    for name, param in model.named_parameters():
        if any(pattern in name for pattern in patterns):
            param.requires_grad = False
            frozen_count += 1
    print(f"Froze first {num_layers} encoder layers ({frozen_count} params)", file=sys.stderr)


def unfreeze_all(model) -> None:
    """Unfreeze all model parameters (used by unfreeze callbacks)."""
    for param in model.parameters():
        param.requires_grad = True
    print("Unfroze all model parameters", file=sys.stderr)
