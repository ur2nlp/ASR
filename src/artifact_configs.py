"""Artifact configuration tracking for reproducibility.

Each pipeline stage saves its configuration to YAML, and subsequent runs
verify that cached artifacts match the current config. Mismatches produce
clear error messages with remediation instructions.

The base layer -- `ArtifactConfig`, `dict_diff`, `format_number` -- is not
defined here. It comes from `lapt-core`, a standalone distribution this
project depends on rather than vendors, so the caching contract cannot drift.
What stays in this module is the part that is genuinely ASR's: the concrete
config classes, and the audio-specific cache naming they depend on.

`ArtifactConfig`, `dict_diff` and `format_number` are re-exported so that
existing `from src.artifact_configs import ...` call sites keep working.
"""

import os
import sys

from dataclasses import asdict, dataclass, field
from lapt_core.artifacts import ArtifactConfig, config_digest, dict_diff, format_number
from omegaconf import DictConfig, OmegaConf

__all__ = [
    "ArtifactConfig",
    "FeatureCacheConfig",
    "FocusTokenizerConfig",
    "ModelConfig",
    "dict_diff",
    "format_number",
    "warn_on_legacy_processed_cache",
]


# ---------------------------------------------------------------------------
# Concrete config classes
# ---------------------------------------------------------------------------

@dataclass
class FeatureCacheConfig(ArtifactConfig):
    """Tracks the extracted-feature cache, shared across models and splits.

    Feature extraction is the expensive stage, and its output depends only on
    the audio and the feature extractor -- not on the tokenizer, the text
    normalization, the train/dev/test assignment, or the length filters. So the
    cache holds features for *every* example, and everything that varies by run
    (labels, splits, filters) is applied afterwards by index, without rewriting
    it. Two checkpoints with the same feature extractor (Whisper small and
    medium both use 80 mel bins) share one cache; so does every FOCUS
    vocabulary and every cross-validation fold.

    Attributes:
        source: The untokenized source record (`source_config_record`).
        feature_extractor: The extractor's own `to_dict()`, minus private keys.
        input_column: Column the features are stored under.
        sampling_rate: Rate the audio is resampled to before extraction.
        source_splits: The source's split names, in storage order.
    """

    artifact_name = "Feature Cache"

    source: dict
    feature_extractor: dict
    input_column: str
    sampling_rate: int
    source_splits: list[str] = field(default_factory=list)

    @classmethod
    def from_args(cls, args: DictConfig, feature_extractor) -> "FeatureCacheConfig":
        from src.processors import get_model_spec
        from src.sources import source_config_record

        extractor_settings = {
            key: value
            for key, value in sorted(feature_extractor.to_dict().items())
            if not key.startswith("_")
        }
        return cls(
            source=source_config_record(args.dataset, args.seed),
            feature_extractor=extractor_settings,
            input_column=get_model_spec(args.model.type).input_column,
            sampling_rate=args.audio.sampling_rate,
        )

    def dirname(self) -> str:
        """Name the cache directory: `features_<extractor>_<digest>`.

        The digest covers everything except `source_splits`, which is only
        known once the source has been read and is implied by `source` anyway.
        """
        keyed = self.to_dict()
        del keyed["source_splits"]
        extractor_type = self.feature_extractor.get("feature_extractor_type", "features")
        return f"features_{_slugify(extractor_type)}_{config_digest(keyed)}"

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
# Cache naming
# ---------------------------------------------------------------------------

def _slugify(value: str) -> str:
    """Reduce an arbitrary config string to a filesystem-safe token."""
    safe = [character if character.isalnum() else "-" for character in str(value)]
    return "".join(safe).strip("-").lower()


def warn_on_legacy_processed_cache(cache_dir: str) -> None:
    """Point out processed caches in a layout that is no longer read.

    Processed data used to live at `<cache_dir>/processed`, then in
    per-configuration `processed_<model>_<settings>` subcaches holding features
    and labels together. Both are superseded by the shared `features_*` cache,
    so say so rather than let them sit taking up disk with no explanation.
    """
    if not os.path.isdir(cache_dir):
        return
    legacy = sorted(
        name for name in os.listdir(cache_dir)
        if (name == "processed" or name.startswith("processed_"))
        and os.path.isdir(os.path.join(cache_dir, name))
    )
    if not legacy:
        return

    print(
        f"Note: found {len(legacy)} processed cache(s) in an old layout under "
        f"{cache_dir}:\n"
        + "".join(f"      {name}\n" for name in legacy)
        + "      Features now live in a shared features_<extractor>_<digest> cache "
        "and labels\n      are encoded per run, so these are no longer read and "
        "can be deleted.",
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
