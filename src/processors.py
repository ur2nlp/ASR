"""ModelSpec registry, tokenizer construction, and processor construction.

`ModelSpec` is the single place where architecture differences are declared.
It captures two axes:

  * **Input representation** — wav2vec2/HuBERT consume raw waveform
    (`input_values` + `Wav2Vec2FeatureExtractor`), W2V-BERT and Whisper consume
    spectral features (`input_features` + `SeamlessM4TFeatureExtractor` /
    `WhisperFeatureExtractor`).
  * **Training objective** — CTC models emit per-frame logits over a character
    vocabulary built from the training data, while seq2seq models decode
    autoregressively over the pretrained checkpoint's own subword vocabulary.

Everything downstream (collator, trainer class, metric decoding, freezing) is
selected from spec fields rather than by branching on the model type string.
"""

import os
import sys
from dataclasses import dataclass, field

from omegaconf import DictConfig
from transformers import (
    HubertForCTC,
    ProcessorMixin,
    SeamlessM4TFeatureExtractor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
    Wav2Vec2BertForCTC,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizerFast,
)

from src import focus
from src.collators import (
    DataCollatorCTCWithPadding,
    DataCollatorSpeechSeq2SeqWithPadding,
)
from src.artifact_configs import FocusTokenizerConfig
from src.vocab import create_ctc_tokenizer, generate_vocab, save_vocab


@dataclass(frozen=True)
class ModelSpec:
    """Captures architecture-specific types and configuration.

    Attributes:
        model_class: The HuggingFace model class to fine-tune.
        feature_extractor_class: The feature extractor class to use.
        tokenizer_class: The tokenizer class to use.
        processor_class: The processor class that combines feature extractor
            and tokenizer.
        input_column: Name of the model input key ("input_values" or
            "input_features").
        objective: Training objective, "ctc" or "seq2seq". Selects which model
            config overrides are applied in `models.py`.
        tokenizer_source: Where the tokenizer comes from. "vocab" builds a
            character tokenizer from the training transcripts; "pretrained"
            loads the checkpoint's own subword tokenizer.
        feature_extractor_source: "config" constructs the feature extractor
            from the `audio` config section; "pretrained" loads it from the
            checkpoint (required when the front end has learned or
            checkpoint-specific parameters, e.g. Whisper's mel filterbank).
        collator_class: The data collator to batch with.
        trainer_class: The HuggingFace Trainer class to train with.
        training_arguments_class: The TrainingArguments class to build.
        feature_extractor_param_patterns: Parameter-name substrings that
            identify the audio front end, used by the freezing helpers.
        encoder_layer_param_pattern: Format string matching a single encoder
            layer's parameters; `{index}` is substituted with the layer index.
        learned_feature_extractor: Whether the front end has trainable weights
            that consume the *raw* audio. When True, features must be
            pre-extracted with the front end frozen or they go stale. Whisper's
            front end is a deterministic mel spectrogram, so this is False even
            though its conv stack is trainable.
        fixed_length_features: Whether every example is padded to a constant
            input length (Whisper's 30 s window), which makes length-grouped
            batching pointless.
        supports_lm_decoding: Whether pyctcdecode/KenLM shallow fusion applies.
        uses_generation: Whether evaluation decodes autoregressively, which
            requires `predict_with_generate` and disables logit argmax
            preprocessing.
        prediction_decode_kwargs: Extra kwargs for decoding predicted IDs into
            strings during metric computation.
        label_decode_kwargs: Extra kwargs for decoding reference label IDs.
    """

    model_class: type
    feature_extractor_class: type
    tokenizer_class: type
    processor_class: type
    input_column: str
    objective: str
    tokenizer_source: str
    feature_extractor_source: str
    collator_class: type
    trainer_class: type
    training_arguments_class: type
    feature_extractor_param_patterns: tuple[str, ...]
    encoder_layer_param_pattern: str = "encoder.layers.{index}."
    learned_feature_extractor: bool = True
    fixed_length_features: bool = False
    supports_lm_decoding: bool = True
    uses_generation: bool = False
    prediction_decode_kwargs: dict = field(
        default_factory=lambda: {"group_tokens": True}
    )
    label_decode_kwargs: dict = field(
        default_factory=lambda: {"group_tokens": False}
    )


# Shared settings for the three CTC architectures, which differ only in their
# model class, feature extractor, and input column.
_CTC_DEFAULTS = {
    "tokenizer_class": Wav2Vec2CTCTokenizer,
    "processor_class": Wav2Vec2Processor,
    "objective": "ctc",
    "tokenizer_source": "vocab",
    "feature_extractor_source": "config",
    "collator_class": DataCollatorCTCWithPadding,
    "trainer_class": Trainer,
    "training_arguments_class": TrainingArguments,
    "feature_extractor_param_patterns": ("feature_extractor", "feature_projection"),
    "learned_feature_extractor": True,
    "fixed_length_features": False,
    "supports_lm_decoding": True,
    "uses_generation": False,
}


MODEL_SPECS: dict[str, ModelSpec] = {
    "wav2vec2": ModelSpec(
        model_class=Wav2Vec2ForCTC,
        feature_extractor_class=Wav2Vec2FeatureExtractor,
        input_column="input_values",
        **_CTC_DEFAULTS,
    ),
    "hubert": ModelSpec(
        model_class=HubertForCTC,
        feature_extractor_class=Wav2Vec2FeatureExtractor,
        input_column="input_values",
        **_CTC_DEFAULTS,
    ),
    "w2v_bert": ModelSpec(
        model_class=Wav2Vec2BertForCTC,
        feature_extractor_class=SeamlessM4TFeatureExtractor,
        input_column="input_features",
        **_CTC_DEFAULTS,
    ),
    "whisper": ModelSpec(
        model_class=WhisperForConditionalGeneration,
        feature_extractor_class=WhisperFeatureExtractor,
        tokenizer_class=WhisperTokenizerFast,
        processor_class=WhisperProcessor,
        input_column="input_features",
        objective="seq2seq",
        tokenizer_source="pretrained",
        feature_extractor_source="pretrained",
        collator_class=DataCollatorSpeechSeq2SeqWithPadding,
        trainer_class=Seq2SeqTrainer,
        training_arguments_class=Seq2SeqTrainingArguments,
        # Whisper's audio front end is the two conv layers of the encoder; the
        # mel spectrogram ahead of them has no learned parameters.
        feature_extractor_param_patterns=("encoder.conv1", "encoder.conv2"),
        learned_feature_extractor=False,
        # every clip is padded or truncated to a 30 s / 3000-frame window
        fixed_length_features=True,
        supports_lm_decoding=False,
        uses_generation=True,
        prediction_decode_kwargs={"skip_special_tokens": True},
        label_decode_kwargs={"skip_special_tokens": True},
    ),
}


# Maps the `model_type` field HuggingFace writes into a checkpoint's
# config.json onto our MODEL_SPECS keys, so that tools operating on a saved
# checkpoint can recover its spec without being told the architecture.
HF_MODEL_TYPE_TO_SPEC_KEY: dict[str, str] = {
    "wav2vec2": "wav2vec2",
    "hubert": "hubert",
    "wav2vec2-bert": "w2v_bert",
    "whisper": "whisper",
}


def get_model_spec(model_type: str) -> ModelSpec:
    """Look up the ModelSpec for a given model type string.

    Args:
        model_type: One of the keys in MODEL_SPECS.

    Returns:
        The corresponding ModelSpec.

    Raises:
        ValueError: If model_type is not recognized.
    """
    if model_type not in MODEL_SPECS:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: {list(MODEL_SPECS.keys())}"
        )
    return MODEL_SPECS[model_type]


def get_spec_for_checkpoint(model_dir: str) -> tuple[str, ModelSpec]:
    """Recover a checkpoint's ModelSpec from its saved config.

    Lets the inference tools load any checkpoint this framework produced (and
    most it did not) without being told which architecture it holds.

    Args:
        model_dir: Path to a saved checkpoint directory, or a Hub model id.

    Returns:
        A `(model_type, spec)` tuple, where `model_type` is our MODEL_SPECS key.

    Raises:
        ValueError: If the checkpoint's architecture has no ModelSpec.
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_dir)
    hf_model_type = getattr(config, "model_type", None)

    if hf_model_type not in HF_MODEL_TYPE_TO_SPEC_KEY:
        raise ValueError(
            f"Checkpoint at {model_dir} has architecture '{hf_model_type}', which "
            f"this framework has no ModelSpec for. "
            f"Supported: {sorted(HF_MODEL_TYPE_TO_SPEC_KEY.keys())}"
        )

    model_type = HF_MODEL_TYPE_TO_SPEC_KEY[hf_model_type]
    return model_type, get_model_spec(model_type)


# ---------------------------------------------------------------------------
# Tokenizer construction
# ---------------------------------------------------------------------------

def setup_tokenizer(
    args: DictConfig,
    train_texts: list[str] | None = None,
    vocab_dir: str | None = None,
    cache_dir: str | None = None,
):
    """Build the tokenizer appropriate to the configured architecture.

    CTC models get a character tokenizer generated from the training
    transcripts (cached as `vocab.json` under `vocab_dir`). Seq2seq models
    reuse the pretrained checkpoint's subword tokenizer, since their decoder
    embeddings are tied to it -- unless `focus.enabled` is set, in which case a
    fresh subword vocabulary is trained over the transcripts and the pretrained
    embeddings are transferred onto it (see `src/focus.py`).

    Args:
        args: Full Hydra config (needs `args.model.*`).
        train_texts: Normalized training transcripts. Required only when the
            spec's `tokenizer_source` is "vocab" and no cached vocab exists, or
            when FOCUS is enabled and its corpus has not been written yet.
        vocab_dir: Directory holding (or receiving) `vocab.json`. Required only
            when `tokenizer_source` is "vocab".
        cache_dir: The dataset cache directory. Required only when FOCUS is
            enabled, which roots its artifacts under `<cache_dir>/focus/`.

    Returns:
        A tokenizer instance.

    Raises:
        ValueError: If required arguments for the tokenizer source are missing,
            or if the spec declares an unknown tokenizer source.
    """
    spec = get_model_spec(args.model.type)

    if focus.is_enabled(args):
        return _setup_focus_tokenizer(args, spec, train_texts, cache_dir)

    if spec.tokenizer_source == "vocab":
        if vocab_dir is None:
            raise ValueError(
                f"Model type '{args.model.type}' builds its tokenizer from the "
                f"training vocabulary, so vocab_dir must be provided."
            )
        vocab_path = os.path.join(vocab_dir, "vocab.json")
        if not os.path.exists(vocab_path):
            if train_texts is None:
                raise ValueError(
                    f"No cached vocab at {vocab_path} and no train_texts given "
                    f"to generate one from."
                )
            vocab = generate_vocab(train_texts)
            save_vocab(vocab, vocab_path)
        return create_ctc_tokenizer(vocab_path)

    if spec.tokenizer_source == "pretrained":
        return _load_pretrained_tokenizer(args, spec)

    raise ValueError(
        f"Unknown tokenizer_source '{spec.tokenizer_source}' for model type "
        f"'{args.model.type}'. Available: vocab, pretrained"
    )


def _setup_focus_tokenizer(
    args: DictConfig,
    spec: ModelSpec,
    train_texts: list[str] | None,
    cache_dir: str | None,
):
    """Train (or load) a FOCUS tokenizer and configure its prefix tokens.

    FOCUS replaces the checkpoint's subword vocabulary with one learned from the
    target transcripts, so it only makes sense for an architecture that would
    otherwise have used the pretrained vocabulary. A CTC model already builds
    its vocabulary from the training data, which is what FOCUS exists to do.
    """
    if spec.tokenizer_source != "pretrained":
        raise ValueError(
            f"focus.enabled=true is not supported for model type "
            f"'{args.model.type}'. FOCUS replaces a pretrained subword "
            f"vocabulary with one learned from the target transcripts, but this "
            f"architecture already builds its vocabulary from them "
            f"(tokenizer_source='{spec.tokenizer_source}'). Set focus.enabled=false."
        )
    if cache_dir is None:
        raise ValueError(
            "focus.enabled=true requires cache_dir so the FOCUS corpus, "
            "tokenizer, and embedding cache can be placed under it."
        )

    paths = focus.resolve_paths(args, cache_dir)

    tokenizer_cached = os.path.exists(os.path.join(paths.tokenizer_dir, "tokenizer.json"))
    if train_texts is not None:
        focus.prepare_corpus(
            train_texts,
            paths,
            num_samples=args.focus.get("num_samples"),
            seed=args.seed,
        )
    elif not tokenizer_cached:
        raise ValueError(
            f"No cached FOCUS tokenizer at {paths.tokenizer_dir} and no "
            f"train_texts given to train one from."
        )

    config = FocusTokenizerConfig.from_args(args)
    config_path = os.path.join(paths.tokenizer_dir, "focus_config.yaml")
    config.check_cached(config_path)

    tokenizer = focus.build_tokenizer(args, spec, paths)
    config.save(config_path)

    # The prefix tokens are rebuilt against the new ids exactly as they are for
    # a pretrained tokenizer; the special-token block was carried over by name,
    # so `set_prefix_tokens` finds every token it needs.
    _configure_prefix_tokens(args, spec, tokenizer)
    return tokenizer


def _load_pretrained_tokenizer(args: DictConfig, spec: ModelSpec):
    """Load a checkpoint's own tokenizer, configured for language and task.

    `language` and `task` are only meaningful for multilingual seq2seq ASR
    models (Whisper), where they select the special tokens that prefix every
    transcript.
    """
    tokenizer = spec.tokenizer_class.from_pretrained(args.model.pretrained_name)
    _configure_prefix_tokens(args, spec, tokenizer)

    print(
        f"Loaded pretrained tokenizer from {args.model.pretrained_name} "
        f"(vocab_size={len(tokenizer)})",
        file=sys.stderr,
    )
    return tokenizer


def _configure_prefix_tokens(args: DictConfig, spec: ModelSpec, tokenizer) -> None:
    """Pin the language and task tokens that prefix every encoded transcript.

    Passing language/task to `from_pretrained` is NOT sufficient for the fast
    tokenizer: it sets the attributes (so `tokenizer.prefix_tokens` looks
    correct) but the post-processor template that actually wraps encoded text is
    built at construction time and is not rebuilt from those kwargs. The result
    is silently wrong labels, missing the <|lang|> and <|task|> tokens.
    `set_prefix_tokens()` rebuilds the template, so call it explicitly.
    """
    language = resolve_language(args, spec)
    task = args.model.get("task", "transcribe")

    if not hasattr(tokenizer, "set_prefix_tokens"):
        return

    tokenizer.set_prefix_tokens(language=language, task=task)
    _verify_prefix_tokens(tokenizer)
    print(
        f"Prefix tokens configured: language={language}, task={task}",
        file=sys.stderr,
    )


def _verify_prefix_tokens(tokenizer) -> None:
    """Check that encoded text really carries the configured prefix tokens.

    Guards against the silent failure described above: every transcript in the
    cached dataset would be mislabeled, and the mistake would only show up as
    an unexplained WER floor much later.
    """
    expected = list(tokenizer.prefix_tokens)
    encoded = tokenizer("probe").input_ids[: len(expected)]

    if encoded != expected:
        raise RuntimeError(
            f"Tokenizer prefix tokens were not applied to encoded text.\n"
            f"  expected: {tokenizer.convert_ids_to_tokens(expected)}\n"
            f"  actual:   {tokenizer.convert_ids_to_tokens(encoded)}\n"
            f"This means transcripts would be encoded without their language "
            f"and task tokens. It usually indicates a transformers version "
            f"whose set_prefix_tokens() no longer rebuilds the fast "
            f"tokenizer's post-processor template."
        )


DETECT_LANGUAGE = "detect"


def resolve_language(args: DictConfig, spec: ModelSpec) -> str | None:
    """Resolve `model.language` into the value the tokenizer and decoder take.

    Only meaningful for Whisper, where the language token prefixed to every
    transcript is selected by this setting. Returns the language unchanged for
    every other architecture.

    Three cases for Whisper:

      * a supported language code or name — returned as given;
      * the string `"detect"` — returns None, which makes both the tokenizer
        and `generate()` omit a pinned language, after warning about what that
        costs (see below);
      * None — an error, because it is almost always an unset config rather
        than a deliberate choice, and the failure it causes is silent.

    The reason None is not simply passed through: the tokenizer and the decoder
    disagree about what it means. `set_prefix_tokens(language=None)` drops the
    language token from the label prefix entirely, so the model is fine-tuned to
    continue `<|startoftranscript|><|transcribe|><|notimestamps|>`. But at
    generation time `language=None` instead triggers Whisper's language-ID pass
    (`WhisperGenerationMixin._retrieve_init_tokens`), which *inserts* a detected
    language token. The decoder is then evaluated on a prefix it never saw in
    fine-tuning, chosen per batch by a classifier that has no token for the
    target language. Nothing raises; eval WER is just quietly worse and
    non-deterministic.

    Raises:
        ValueError: If the language is None or is not one Whisper knows.
    """
    language = args.model.get("language")

    if spec.tokenizer_class is not WhisperTokenizerFast:
        return language

    from transformers.models.whisper.tokenization_whisper import (
        LANGUAGES,
        TO_LANGUAGE_CODE,
    )

    if language is None:
        raise ValueError(
            "model.language is not set.\n"
            "Whisper prefixes a language token to every transcript, and leaving "
            "it unset does not mean 'no language': the tokenizer omits the token "
            "from training labels while generate() detects and inserts one "
            "anyway, so the decoder is evaluated on a prefix it never trained "
            "on.\n"
            "Set model.language to one of:\n"
            "  * a supported language code or name — if your target language is "
            "not among them, a close proxy is the usual choice, since "
            "fine-tuning re-maps whatever token you pick;\n"
            f"  * '{DETECT_LANGUAGE}' to deliberately accept the mismatch above "
            "and let Whisper detect the language at generation time.\n"
            f"Supported codes: {sorted(LANGUAGES.keys())}"
        )

    if str(language).lower() == DETECT_LANGUAGE:
        _warn_language_detection()
        return None

    normalized = str(language).lower()
    if normalized in LANGUAGES or normalized in TO_LANGUAGE_CODE:
        return language

    raise ValueError(
        f"Whisper has no language token for '{language}'. It was pretrained on "
        f"{len(LANGUAGES)} languages and cannot represent others directly.\n"
        f"Options:\n"
        f"  * set model.language to a close proxy language whose token you are "
        f"willing to repurpose (fine-tuning will re-map it to your language);\n"
        f"  * set model.language to '{DETECT_LANGUAGE}' to let Whisper detect "
        f"the language at generation time, accepting the train/eval prefix "
        f"mismatch that implies.\n"
        f"Supported codes: {sorted(LANGUAGES.keys())}"
    )


def _warn_language_detection() -> None:
    """Warn that language detection was deliberately opted into."""
    print(
        "\n"
        "  WARNING: model.language='detect' — no language token is pinned.\n"
        "  Training labels will be prefixed with\n"
        "      <|startoftranscript|><|transcribe|><|notimestamps|>\n"
        "  but generate() runs Whisper's language-ID pass and prefixes\n"
        "      <|startoftranscript|><|LANG|><|transcribe|><|notimestamps|>\n"
        "  with <|LANG|> chosen per batch from the audio. The decoder is "
        "therefore\n"
        "  evaluated on a prefix distribution it never saw during fine-tuning, "
        "and\n"
        "  eval WER will be both degraded and non-deterministic across runs.\n"
        "  Pin a proxy language instead unless you are deliberately measuring "
        "this.\n",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Processor construction
# ---------------------------------------------------------------------------

def setup_processor(
    args: DictConfig,
    tokenizer,
) -> ProcessorMixin:
    """Construct a processor by pairing a tokenizer with the appropriate
    feature extractor for the configured model architecture.

    Args:
        args: Full Hydra config (needs args.model.type and args.audio).
        tokenizer: The tokenizer produced by `setup_tokenizer`.

    Returns:
        A processor combining the feature extractor and tokenizer.
    """
    spec = get_model_spec(args.model.type)
    feature_extractor = _build_feature_extractor(args, spec)

    processor = spec.processor_class(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )

    print(
        f"Built processor: {spec.processor_class.__name__} "
        f"(input_column={spec.input_column})",
        file=sys.stderr,
    )
    return processor


def _build_feature_extractor(args: DictConfig, spec: ModelSpec):
    """Build the feature extractor from config or load it from the checkpoint.

    Checkpoint-sourced extractors (Whisper) carry parameters the model depends
    on exactly — number of mel bins, FFT size, chunk length — so constructing
    one from our generic `audio` config section would silently produce inputs
    the encoder cannot consume.
    """
    if spec.feature_extractor_source == "pretrained":
        feature_extractor = spec.feature_extractor_class.from_pretrained(
            args.model.pretrained_name
        )
        _warn_on_sampling_rate_mismatch(args, feature_extractor)
        return feature_extractor

    if spec.feature_extractor_source == "config":
        return spec.feature_extractor_class(
            feature_size=args.audio.feature_size,
            sampling_rate=args.audio.sampling_rate,
            padding_value=0.0,
            do_normalize=args.audio.do_normalize,
            return_attention_mask=args.audio.return_attention_mask,
        )

    raise ValueError(
        f"Unknown feature_extractor_source '{spec.feature_extractor_source}' "
        f"for model type '{args.model.type}'. Available: config, pretrained"
    )


def _warn_on_sampling_rate_mismatch(args: DictConfig, feature_extractor) -> None:
    """Warn when the audio config resamples to a rate the extractor rejects."""
    expected = getattr(feature_extractor, "sampling_rate", None)
    configured = args.audio.sampling_rate
    if expected is not None and expected != configured:
        print(
            f"Warning: audio.sampling_rate={configured} but "
            f"{type(feature_extractor).__name__} expects {expected}. "
            f"Set audio.sampling_rate={expected} to avoid distorted features.",
            file=sys.stderr,
        )
