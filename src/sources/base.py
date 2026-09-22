"""Registry and shared base for the audio source artifacts.

Each dataset type is a `DatasetArtifact` subclass that knows how to read its own
configuration and build its own contents. The registry maps a config's `type`
field to the class, replacing the `if/elif` chain the dispatcher used to carry.

Registration is an explicit call per type module rather than an
`__init_subclass__` hook: a linter cannot delete it, a test subclass does not
pollute a global table, and the set of live types is greppable.
"""

from datasets import DatasetDict
from lapt_core.dataset_artifacts import DatasetArtifact, DatasetRegistry

SOURCE_TYPES = DatasetRegistry()


class AudioSourceArtifact(DatasetArtifact):
    """A speech corpus stage producing standardized `{audio, transcription}`.

    Subclasses implement `load`, which returns whatever shape the underlying
    source has; this class renames the audio and transcription columns to the
    names the rest of the pipeline expects, and it does so *before* the dataset
    is written, so the cached copy is already standardized.

    The column names are therefore part of `config()`: changing one changes the
    stored contents. Text normalization is deliberately *not* -- it runs after
    this stage and is not persisted here, so recording it would invalidate a
    cache it cannot affect.
    """

    type_name: str | None = None

    def __init__(self, cache_dir: str, audio_column: str = "audio",
                 text_column: str = "transcription"):
        """Initialize the source.

        Args:
            cache_dir: Directory the `untokenized` subdirectory is created in.
            audio_column: Name of the audio column in the source data.
            text_column: Name of the transcription column in the source data.
        """
        super().__init__(cache_dir)
        self.audio_column = audio_column
        self.text_column = text_column

    def load(self) -> DatasetDict:
        """Return the source's contents, before column standardization."""
        raise NotImplementedError

    def column_config(self) -> dict:
        """Return the column-renaming half of `config()`, for subclasses to merge."""
        return {"audio_column": self.audio_column, "text_column": self.text_column}

    def build(self, deps) -> DatasetDict:
        """Load the source and standardize its column names.

        Args:
            deps: Unused; these sources take their inputs through the
                constructor rather than from an `ArtifactGraph`.

        Returns:
            A `DatasetDict` whose splits carry `audio` and `transcription`.
        """
        return self._standardize(self.load())

    def _standardize(self, dataset: DatasetDict) -> DatasetDict:
        """Rename the configured columns to `audio` and `transcription`."""
        return standardize_columns(dataset, self.audio_column, self.text_column)

    @classmethod
    def from_config(cls, cache_dir: str, source_config, seed: int = 1) -> "AudioSourceArtifact":
        """Construct from a dataset configuration entry."""
        raise NotImplementedError


def standardize_columns(
    dataset: DatasetDict,
    audio_column: str = "audio",
    text_column: str = "transcription",
) -> DatasetDict:
    """Rename a dataset's audio and transcription columns to the standard names.

    A column that is already standard, or absent from a split, is left alone,
    so this is safe to apply to a dataset that has been standardized already.

    Args:
        dataset: The dataset to rename columns in.
        audio_column: Current name of the audio column.
        text_column: Current name of the transcription column.

    Returns:
        The same dataset, with columns renamed in place.
    """
    for split_name in dataset:
        split = dataset[split_name]
        if audio_column != "audio" and audio_column in split.column_names:
            dataset[split_name] = split.rename_column(audio_column, "audio")
        if (
            text_column != "transcription"
            and text_column in dataset[split_name].column_names
        ):
            dataset[split_name] = dataset[split_name].rename_column(
                text_column, "transcription"
            )
    return dataset
