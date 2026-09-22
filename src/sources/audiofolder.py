"""HuggingFace `audiofolder` corpus: audio files plus a metadata manifest."""

from datasets import DatasetDict, load_dataset

from src.sources.base import SOURCE_TYPES, AudioSourceArtifact


class AudiofolderDataset(AudioSourceArtifact):
    """A corpus read by the `datasets` audiofolder builder."""

    type_name = "audiofolder"

    def __init__(self, cache_dir: str, path: str, **columns):
        """Initialize the source.

        Args:
            cache_dir: Directory the `untokenized` subdirectory goes in.
            path: Directory the audiofolder builder reads.
            **columns: `audio_column` / `text_column`, see `AudioSourceArtifact`.
        """
        super().__init__(cache_dir, **columns)
        self.source_path = path

    def config(self) -> dict:
        """Return the parameters this cache is keyed on."""
        return {"type": "audiofolder", "path": self.source_path, **self.column_config()}

    def load(self) -> DatasetDict:
        """Load the directory through the audiofolder builder."""
        return load_dataset("audiofolder", data_dir=self.source_path)

    @classmethod
    def from_config(
        cls, cache_dir: str, source_config, seed: int = 1
    ) -> "AudiofolderDataset":
        """Construct from a dataset configuration entry.

        Args:
            cache_dir: Directory the `untokenized` subdirectory goes in.
            source_config: Entry carrying `path`.
            seed: Unused; nothing here is random.

        Returns:
            The configured source.
        """
        get = source_config.get
        return cls(
            cache_dir,
            path=source_config["path"],
            audio_column=get("audio_column", "audio"),
            text_column=get("text_column", "transcription"),
        )


SOURCE_TYPES.register(AudiofolderDataset)
