"""Several sources concatenated into one corpus."""

from lapt_core.composites import ConcatArtifact

from src.sources.base import SOURCE_TYPES


class AudioConcatDataset(ConcatArtifact):
    """Concatenate sources, preserving every split they carry.

    Everything comes from `lapt_core`: the child factory seam, the per-child
    cache directories keyed by source id, the `config()` record covering the
    whole child tree, and the split-by-split concatenation.

    That last one used to be overridden here. The shared `build` took each
    child's `train` and returned a single-split result, which suited a corpus
    of undifferentiated text but silently discarded held-out data for speech
    corpora, which routinely arrive pre-split -- a Hub dataset with
    train/validation/test, a paired directory with train/ and test/
    subdirectories. Being the first thing a second implementation found wrong
    with that interface made it a defect in the shared layer rather than a
    local need, so the behaviour moved upstream in lapt-core 0.1.1 and this
    class is now a registration shim.
    """

    type_name = "concat"

    @classmethod
    def from_config(
        cls, cache_dir: str, source_config, seed: int = 1
    ) -> "AudioConcatDataset":
        """Construct from a dataset configuration entry.

        Args:
            cache_dir: Directory the `untokenized` subdirectory goes in, and
                the parent of each child's own cache directory.
            source_config: Entry carrying `sources`.
            seed: Global random seed, passed down to children.

        Returns:
            The configured composite.
        """
        from src.sources.factory import make_source, normalize_sources

        return cls(
            cache_dir,
            normalize_sources(source_config.get("sources")),
            child_factory=make_source,
            parent_id=source_config.get("id"),
            seed=seed,
        )


SOURCE_TYPES.register(AudioConcatDataset)
