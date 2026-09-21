"""Artifact configuration tracking for reproducibility.

Each pipeline stage saves its configuration to YAML, and subsequent runs
verify that cached artifacts match the current config. Mismatches produce
clear error messages with remediation instructions.

The base layer -- `ArtifactConfig`, `dict_diff`, `format_number` -- is not
defined here. It comes from `lapt-core`, a standalone distribution shared with
the LAPT repository, so the two projects cannot drift apart the way a
hand-copied version did. What stays in this module is the part that is
genuinely ASR's: the concrete config classes, and the audio-specific cache
naming they depend on.

`ArtifactConfig`, `dict_diff` and `format_number` are re-exported so that
existing `from src.artifact_configs import ...` call sites keep working.
"""

import os
import sys

from dataclasses import asdict, dataclass, field
from lapt_core.artifacts import ArtifactConfig, dict_diff, format_number
from omegaconf import DictConfig, OmegaConf

__all__ = [
    "ArtifactConfig",
    "DatasetConfig",
    "FocusTokenizerConfig",
    "ModelConfig",
    "ProcessedDatasetConfig",
    "dict_diff",
    "format_number",
    "processed_cache_dirname",
    "warn_on_legacy_processed_cache",
]


# ---------------------------------------------------------------------------
# Concrete config classes
# ---------------------------------------------------------------------------

class DatasetConfig(ArtifactConfig):
    """Tracks configuration for the untokenized dataset stage."""

    artifact_name = "Untokenized Dataset"

    def __init__(self, config: dict):
        self._config = config

    @classmethod
    def from_args(cls, args: DictConfig) -> "DatasetConfig":
        config = {
            "type": args.dataset.type,
            "id": args.dataset.id,
            "seed": args.seed,
        }

        dataset_type = args.dataset.type
        if dataset_type == "paired":
            config["path"] = args.dataset.path
            config["audio_ext"] = args.dataset.get("audio_ext", ".wav")
            config["transcript_ext"] = args.dataset.get("transcript_ext", ".txt")
            config["recursive"] = args.dataset.get("recursive", False)
        elif dataset_type == "audiofolder":
            config["path"] = args.dataset.path
        elif dataset_type == "huggingface":
            config["name"] = args.dataset.name
            config["config"] = getattr(args.dataset, "config", None)
        elif dataset_type == "concat":
            config["sources"] = OmegaConf.to_container(
                args.dataset.sources, resolve=True
            )

        # include preprocessing config since it affects cached data
        config["preprocessing"] = OmegaConf.to_container(
            args.preprocessing, resolve=True
        )

        return cls(config)

    def to_dict(self) -> dict:
        return dict(self._config)


@dataclass
class ProcessedDatasetConfig(ArtifactConfig):
    """Tracks configuration for the processed (feature-extracted) dataset."""

    artifact_name = "Processed Dataset"

    model_type: str
    pretrained_name: str
    sampling_rate: int
    max_audio_length_seconds: float | None
    vocab_size: int
    max_label_length: int | None = None
    # Only meaningful for multilingual seq2seq models, where they change the
    # special tokens prefixed to every encoded transcript and therefore the
    # cached labels.
    language: str | None = None
    task: str | None = None
    # Identifies the FOCUS tokenizer when vocabulary replacement is on. A
    # different FOCUS vocabulary means entirely different cached labels, so it
    # must not share a subcache with the pretrained-vocabulary run.
    focus_tokenizer_id: str | None = None

    @classmethod
    def from_args(cls, args: DictConfig, vocab_size: int) -> "ProcessedDatasetConfig":
        from src.focus import is_enabled as focus_is_enabled, tokenizer_id

        return cls(
            model_type=args.model.type,
            pretrained_name=args.model.pretrained_name,
            sampling_rate=args.audio.sampling_rate,
            max_audio_length_seconds=args.dataset.get("max_audio_length_seconds"),
            vocab_size=vocab_size,
            max_label_length=args.dataset.get("max_label_length"),
            language=args.model.get("language"),
            task=args.model.get("task"),
            focus_tokenizer_id=tokenizer_id(args) if focus_is_enabled(args) else None,
        )

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# FOCUS vocabulary replacement
# ---------------------------------------------------------------------------

@dataclass
class FocusTokenizerConfig(ArtifactConfig):
    """Tracks the settings that determine the FOCUS tokenizer artifact.

    Only the vocabulary-shaping fields live here. The FOCUS *embedding* knobs
    (the fastText hyperparameters) do not change the tokenizer and are keyed
    separately by `focus.embedding_hash`, so tuning them reuses the tokenizer.
    """

    artifact_name = "FOCUS Tokenizer"

    pretrained_name: str
    vocab_size: int
    tokenizer_algorithm: str | None
    character_coverage: float
    num_samples: int | None
    seed: int

    @classmethod
    def from_args(cls, args: DictConfig) -> "FocusTokenizerConfig":
        return cls(
            pretrained_name=args.model.pretrained_name,
            vocab_size=args.focus.vocab_size,
            tokenizer_algorithm=args.focus.get("tokenizer_algorithm"),
            character_coverage=args.focus.get("character_coverage", 1.0),
            num_samples=args.focus.get("num_samples"),
            seed=args.seed,
        )

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Processed-cache naming
# ---------------------------------------------------------------------------

def _format_sampling_rate(sampling_rate: int) -> str:
    """Render a sampling rate compactly: 16000 -> '16k'."""
    if sampling_rate % 1000 == 0:
        return f"{sampling_rate // 1000}k"
    return str(sampling_rate)


def _format_seconds(seconds: float) -> str:
    """Render a duration without a trailing '.0': 30.0 -> '30'."""
    if float(seconds).is_integer():
        return str(int(seconds))
    return str(seconds).replace(".", "p")


def _slugify(value: str) -> str:
    """Reduce an arbitrary config string to a filesystem-safe token."""
    safe = [character if character.isalnum() else "-" for character in str(value)]
    return "".join(safe).strip("-").lower()


def processed_cache_dirname(args: DictConfig) -> str:
    """Name the processed-dataset subcache for this run's configuration.

    Feature extraction and label encoding are model-specific -- a wav2vec2 cache
    holds raw-waveform `input_values` and character labels, a Whisper cache holds
    mel `input_features` and subword labels -- so they cannot share a directory.
    Rather than one `processed/` that every run invalidates in turn, each
    configuration gets its own sibling subcache and they coexist:

        data/zulu/untokenized/                            (text, model-agnostic)
        data/zulu/vocab/                                  (CTC char vocab)
        data/zulu/processed_xlsr300m_sr16k_a30/
        data/zulu/processed_whisper_small_sr16k_a30_l448_af-transcribe/

    The name carries only the fields derivable from `args`, so it can be
    computed before the dataset is loaded (the `fresh_processed` cleanup needs
    it). It is a readable key, not a complete fingerprint: the full tracked set,
    including the vocab size that is only known after stage 1, lives in the
    `config.yaml` written inside the subcache, and `ProcessedDatasetConfig`
    still errors on a mismatch within a directory.

    Args:
        args: Full Hydra config (needs args.model, args.audio, args.dataset).

    Returns:
        Directory name such as `processed_whisper_small_sr16k_a30_l448_af-transcribe`.
    """
    model_short = args.model.get("short_name") or args.model.type
    parts = [_slugify(model_short), f"sr{_format_sampling_rate(args.audio.sampling_rate)}"]

    max_audio_length = args.dataset.get("max_audio_length_seconds")
    if max_audio_length is not None:
        parts.append(f"a{_format_seconds(max_audio_length)}")

    max_label_length = args.dataset.get("max_label_length")
    if max_label_length is not None:
        parts.append(f"l{int(max_label_length)}")

    # Only multilingual seq2seq models prefix language/task tokens to their
    # labels; for everything else these are absent and add nothing to the name.
    language = args.model.get("language")
    task = args.model.get("task")
    if language is not None:
        parts.append(f"{_slugify(language)}-{_slugify(task)}" if task else _slugify(language))

    # A replaced vocabulary changes every cached label, so it gets its own
    # subcache beside the pretrained-vocabulary one rather than invalidating it.
    from src.focus import is_enabled as focus_is_enabled, tokenizer_id

    if focus_is_enabled(args):
        parts.append(_slugify(tokenizer_id(args)))

    return "processed_" + "_".join(parts)


def warn_on_legacy_processed_cache(cache_dir: str) -> None:
    """Point out a pre-subcache `processed/` directory left by an older run.

    Processed data used to live at a single `<cache_dir>/processed`. Those
    directories are no longer read, so say so rather than let them sit taking up
    disk with no explanation.
    """
    legacy_path = os.path.join(cache_dir, "processed")
    if not os.path.isdir(legacy_path):
        return

    print(
        f"Note: found a processed cache in the old layout at {legacy_path}.\n"
        f"      Processed data now lives in per-configuration subcaches named\n"
        f"      processed_<model>_<settings>, so this directory is no longer "
        f"read and can be deleted.",
        file=sys.stderr,
    )


@dataclass
class ModelConfig(ArtifactConfig):
    """Tracks the full model + training configuration."""

    artifact_name = "Model"

    model_type: str
    pretrained_name: str
    vocab_size: int
    training_config: dict = field(default_factory=dict)

    @classmethod
    def from_args(cls, args: DictConfig, vocab_size: int) -> "ModelConfig":
        return cls(
            model_type=args.model.type,
            pretrained_name=args.model.pretrained_name,
            vocab_size=vocab_size,
            training_config=OmegaConf.to_container(args.training, resolve=True),
        )

    def to_dict(self) -> dict:
        return asdict(self)
