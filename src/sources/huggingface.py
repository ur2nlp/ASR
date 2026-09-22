"""A corpus pulled from the HuggingFace Hub."""

from datasets import Dataset, DatasetDict, load_dataset

from src.sources.base import SOURCE_TYPES, AudioSourceArtifact


class HuggingFaceDataset(AudioSourceArtifact):
    """A Hub dataset, optionally restricted to one subset or split.

    Note on naming: `load_dataset`'s `path` argument is the Hub identifier and
    its `name` argument is the subset. The configuration calls these `name` and
    `config` instead, to avoid calling an identifier a "path".
    """

    type_name = "huggingface"

    def __init__(
        self,
        cache_dir: str,
        name: str,
        config_name: str | None = None,
        split: str | None = None,
        trust_remote_code: bool | None = None,
        **columns,
    ):
        """Initialize the source.

        Args:
            cache_dir: Directory the `untokenized` subdirectory goes in.
            name: Hub dataset identifier.
            config_name: Subset name, passed to `load_dataset` as `name`.
            split: Single split to load, if not the whole dataset.
            trust_remote_code: Whether the dataset's own loading script may run.
            **columns: `audio_column` / `text_column`, see `AudioSourceArtifact`.
        """
        super().__init__(cache_dir, **columns)
        self.name = name
        self.config_name = config_name
        self.split = split
        self.trust_remote_code = trust_remote_code

    def config(self) -> dict:
        """Return the parameters this cache is keyed on.

        `trust_remote_code` is included because it decides whether the dataset's
        own script runs, which can change what is loaded -- not merely how.
        """
        return {
            "type": "huggingface",
            "name": self.name,
            "config": self.config_name,
            "split": self.split,
            "trust_remote_code": self.trust_remote_code,
            **self.column_config(),
        }

    def load(self) -> DatasetDict:
        """Fetch the dataset, wrapping a single split into a `DatasetDict`."""
        kwargs = {"path": self.name}
        if self.config_name is not None:
            kwargs["name"] = self.config_name
        if self.split is not None:
            kwargs["split"] = self.split
        if self.trust_remote_code is not None:
            kwargs["trust_remote_code"] = self.trust_remote_code

        result = load_dataset(**kwargs)
        if isinstance(result, Dataset):
            return DatasetDict({"train": result})
        return result

    @classmethod
    def from_config(
        cls, cache_dir: str, source_config, seed: int = 1
    ) -> "HuggingFaceDataset":
        """Construct from a dataset configuration entry.

        Args:
            cache_dir: Directory the `untokenized` subdirectory goes in.
            source_config: Entry carrying `name` and optional `config`/`split`.
            seed: Unused; this source does not subsample.

        Returns:
            The configured source.
        """
        get = source_config.get
        return cls(
            cache_dir,
            name=source_config["name"],
            config_name=get("config"),
            split=get("split"),
            trust_remote_code=get("trust_remote_code"),
            audio_column=get("audio_column", "audio"),
            text_column=get("text_column", "transcription"),
        )


SOURCE_TYPES.register(HuggingFaceDataset)
