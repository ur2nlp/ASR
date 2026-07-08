"""Artifact configuration tracking for reproducibility.

Ports LAPT's ArtifactConfig pattern: each pipeline stage saves its
configuration to YAML, and subsequent runs verify that cached artifacts
match the current config. Mismatches produce clear error messages with
remediation instructions.
"""

import os
import sys

import yaml
from dataclasses import asdict, dataclass, field
from omegaconf import DictConfig, OmegaConf


# ---------------------------------------------------------------------------
# Recursive dict comparison
# ---------------------------------------------------------------------------

def _dict_diff(dict1: dict, dict2: dict, path: str = "") -> list[str]:
    """Recursively compare two dicts and return human-readable diffs."""
    diffs = []

    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())

    for key in sorted(keys1 - keys2):
        full_path = f"{path}.{key}" if path else key
        diffs.append(f"{full_path}: present in cached but not in current")

    for key in sorted(keys2 - keys1):
        full_path = f"{path}.{key}" if path else key
        diffs.append(f"{full_path}: present in current but not in cached")

    for key in sorted(keys1 & keys2):
        val1, val2 = dict1[key], dict2[key]
        full_path = f"{path}.{key}" if path else key

        if isinstance(val1, dict) and isinstance(val2, dict):
            diffs.extend(_dict_diff(val1, val2, full_path))
        elif val1 != val2:
            diffs.append(f"{full_path}: {val1} (cached) != {val2} (current)")

    return diffs


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class ArtifactConfig:
    """Base class for artifact configuration tracking."""

    artifact_name: str = "Artifact"

    def to_dict(self) -> dict:
        raise NotImplementedError

    def save(self, config_path: str) -> None:
        """Write config to YAML for later verification."""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        print(f"Saved {self.artifact_name} config to {config_path}", file=sys.stderr)

    def check_cached(self, config_path: str, error_on_mismatch: bool = True) -> bool:
        """Verify that a cached config matches the current one.

        Args:
            config_path: Path to the cached YAML config.
            error_on_mismatch: If True, raise ValueError on mismatch.

        Returns:
            True if configs match or no cached config exists.
        """
        if not os.path.exists(config_path):
            return True

        with open(config_path, "r") as f:
            cached_config = yaml.safe_load(f)

        diffs = _dict_diff(cached_config, self.to_dict())
        if not diffs:
            return True

        diff_str = "\n  ".join(diffs)
        error_msg = (
            f"\n{'=' * 70}\n"
            f"CONFIG MISMATCH: {self.artifact_name}\n"
            f"{'=' * 70}\n"
            f"Differences:\n  {diff_str}\n\n"
            f"The cached artifact at {config_path} was built with different settings.\n"
            f"To rebuild, delete the cache directory or set the appropriate fresh_* flag.\n"
            f"{'=' * 70}"
        )

        if error_on_mismatch:
            raise ValueError(error_msg)

        print(error_msg, file=sys.stderr)
        return False


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

    @classmethod
    def from_args(cls, args: DictConfig, vocab_size: int) -> "ProcessedDatasetConfig":
        return cls(
            model_type=args.model.type,
            pretrained_name=args.model.pretrained_name,
            sampling_rate=args.audio.sampling_rate,
            max_audio_length_seconds=args.dataset.get("max_audio_length_seconds"),
            vocab_size=vocab_size,
        )

    def to_dict(self) -> dict:
        return asdict(self)


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
