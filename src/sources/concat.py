"""Several sources concatenated into one corpus."""

import sys

from datasets import DatasetDict, concatenate_datasets
from lapt_core.composites import ConcatArtifact

from src.sources.base import SOURCE_TYPES


class AudioConcatDataset(ConcatArtifact):
    """Concatenate sources, preserving every split they carry.

    Everything except `build` comes from `lapt_core`: the child factory seam,
    the per-child cache directories keyed by source id, and the `config()`
    record that captures the whole child tree.

    `build` is overridden because the shared default concatenates only the
    `train` split and returns a single-split dataset. That suits a corpus whose
    sources are undifferentiated text, but speech corpora routinely arrive
    pre-split -- a Hub dataset with `train`/`validation`/`test`, a paired
    directory with `train/` and `test/` subdirectories -- and dropping those
    would silently discard held-out data that the config asked for. Splits are
    therefore unioned: each split name present in any source becomes a split of
    the result, concatenated across the sources that have it.
    """

    type_name = "concat"

    def build(self, deps) -> DatasetDict:
        """Resolve each child and concatenate them split by split.

        Args:
            deps: Unused; the child set is known only from the configuration,
                so children are resolved here rather than injected.

        Returns:
            A `DatasetDict` holding every split name any source carried.
        """
        print(f"Concatenating {len(self.sources)} dataset sources", file=sys.stderr)

        splits: dict[str, list] = {}
        for index, (child_id, child) in enumerate(self.children()):
            child_data = child.resolve()
            for split_name, split_data in child_data.items():
                splits.setdefault(split_name, []).append(split_data)
            sizes = ", ".join(
                f"{name}={len(data)}" for name, data in child_data.items()
            )
            print(f"  Source {index} ({child_id}): {sizes}", file=sys.stderr)

        concatenated = DatasetDict({
            split_name: concatenate_datasets(split_list)
            for split_name, split_list in splits.items()
        })
        totals = ", ".join(
            f"{name}={len(data)}" for name, data in concatenated.items()
        )
        print(f"  Concatenated to {totals}", file=sys.stderr)
        return concatenated

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
