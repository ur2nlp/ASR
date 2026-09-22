"""Construction of source artifacts from dataset configuration entries.

Named `make_source` rather than `build_source` because `build()` on an artifact
means something else: this returns an unresolved *object*, while `build()`
produces the dataset contents. Nothing here touches disk.
"""

from typing import Any

from lapt_core.dataset_artifacts import DatasetArtifact
from omegaconf import DictConfig, OmegaConf

from src.sources.base import SOURCE_TYPES


def normalize_sources(sources) -> list[dict]:
    """Unwrap a composite's `sources` list into plain dicts.

    The single omegaconf boundary for composites. `lapt_core.composites` takes
    plain dicts so it need not depend on Hydra; converting here means it happens
    once, at construction, rather than every time a `config()` record is built.

    Args:
        sources: Entries from a configuration, as `DictConfig`, `ListConfig`
            or plain dicts.

    Returns:
        The same entries as plain dicts, with interpolations resolved.
    """
    return [
        OmegaConf.to_container(DictConfig(source), resolve=True)
        for source in (sources or [])
    ]


def as_plain_dict(source_config: Any) -> dict:
    """Return a configuration entry as a plain dict.

    Args:
        source_config: A `DictConfig` or an already-plain mapping.

    Returns:
        A plain dict with interpolations resolved.
    """
    if isinstance(source_config, DictConfig):
        return OmegaConf.to_container(source_config, resolve=True)
    return dict(source_config)


def make_source(
    cache_dir: str,
    source_config: Any,
    seed: int = 1,
) -> DatasetArtifact:
    """Construct the source artifact a configuration entry describes.

    The signature is the `child_factory` seam `lapt_core.composites` expects,
    so a composite can construct its children without knowing this project's
    registry or type names.

    Args:
        cache_dir: Directory the source's `untokenized` subdirectory goes in.
        source_config: The configuration entry, carrying at least `type`.
        seed: Global random seed, passed to sources that sample.

    Returns:
        An unresolved source artifact.

    Raises:
        ValueError: If no source type is registered under the config's `type`.
    """
    config = as_plain_dict(source_config)
    if "type" not in config:
        raise ValueError(
            "Dataset configuration has no 'type' field. "
            f"Known types: {', '.join(SOURCE_TYPES.known_types())}"
        )
    return SOURCE_TYPES.get(config["type"]).from_config(cache_dir, config, seed)


def source_config_record(source_config: Any, seed: int = 1) -> dict:
    """Return the cache-keying record a configuration entry would produce.

    A fingerprint of the dataset, independent of where it is cached: `config()`
    never reads the artifact's root, so a placeholder is passed for it. Used by
    callers that need to know *what data* a run uses without resolving it.

    Args:
        source_config: The configuration entry, carrying at least `type`.
        seed: Global random seed, passed to sources that sample.

    Returns:
        The artifact's `config()` dict.
    """
    return make_source("", source_config, seed).config()
