"""Paired audio/transcript corpus: `utt001.wav` beside `utt001.txt`."""

import glob
import os
import sys

from datasets import Audio, Dataset, DatasetDict

from src.sources.base import SOURCE_TYPES, AudioSourceArtifact


class PairedDataset(AudioSourceArtifact):
    """A corpus of matched audio and transcript files.

    Each audio file is paired with a sibling text file of the same stem holding
    its transcription -- the common layout for field-collected and manually
    transcribed speech.

    Two directory layouts are supported. If `path` contains any of `train/`,
    `dev/` or `test/`, each is loaded as the corresponding split; otherwise
    every pair directly under `path` becomes a single `train` split, and a dev
    split is carved out later via `dev_size`.
    """

    type_name = "paired"

    def __init__(
        self,
        cache_dir: str,
        path: str,
        audio_ext: str = ".wav",
        transcript_ext: str = ".txt",
        recursive: bool = False,
        **columns,
    ):
        """Initialize the source.

        Args:
            cache_dir: Directory the `untokenized` subdirectory goes in.
            path: Root directory holding the pairs, or the split subdirectories.
            audio_ext: Audio file extension to match.
            transcript_ext: Transcript file extension to match.
            recursive: Whether to search subdirectories.
            **columns: `audio_column` / `text_column`, see `AudioSourceArtifact`.
        """
        super().__init__(cache_dir, **columns)
        self.source_path = path
        self.audio_ext = audio_ext
        self.transcript_ext = transcript_ext
        self.recursive = recursive

    def config(self) -> dict:
        """Return the parameters this cache is keyed on."""
        return {
            "type": "paired",
            "path": self.source_path,
            "audio_ext": self.audio_ext,
            "transcript_ext": self.transcript_ext,
            "recursive": self.recursive,
            **self.column_config(),
        }

    def load(self) -> DatasetDict:
        """Read the pairs into one split per split subdirectory, or one `train`.

        Returns:
            A `DatasetDict` with an `audio` column cast to the Audio feature.

        Raises:
            FileNotFoundError: If the path is not a directory, or no complete
                pairs are found.
        """
        if not os.path.isdir(self.source_path):
            raise FileNotFoundError(
                f"Paired dataset path does not exist or is not a directory: {self.source_path}"
            )

        split_subdirs = [
            name for name in ("train", "dev", "test")
            if os.path.isdir(os.path.join(self.source_path, name))
        ]

        if split_subdirs:
            return DatasetDict({
                name: self._build_split(os.path.join(self.source_path, name))
                for name in split_subdirs
            })

        return DatasetDict({"train": self._build_split(self.source_path)})

    def _build_split(self, directory: str) -> Dataset:
        """Build one split from a directory of audio/transcript pairs."""
        return build_paired_split(
            directory, self.audio_ext, self.transcript_ext, self.recursive
        )

    @classmethod
    def from_config(cls, cache_dir: str, source_config, seed: int = 1) -> "PairedDataset":
        """Construct from a dataset configuration entry.

        Args:
            cache_dir: Directory the `untokenized` subdirectory goes in.
            source_config: Entry carrying `path` and optional extensions.
            seed: Unused; nothing here is random.

        Returns:
            The configured source.
        """
        get = source_config.get
        return cls(
            cache_dir,
            path=source_config["path"],
            audio_ext=get("audio_ext", ".wav"),
            transcript_ext=get("transcript_ext", ".txt"),
            recursive=get("recursive", False),
            audio_column=get("audio_column", "audio"),
            text_column=get("text_column", "transcription"),
        )


def build_paired_split(
    directory: str,
    audio_ext: str = ".wav",
    transcript_ext: str = ".txt",
    recursive: bool = False,
) -> Dataset:
    """Build one `Dataset` from a directory of audio/transcript pairs.

    Module-level rather than a method because the external held-out eval sets
    use the same layout without being a cached source: they are loaded fresh
    each run, so they need the reader without the artifact around it.

    Args:
        directory: Directory to search for audio files.
        audio_ext: Audio file extension to match.
        transcript_ext: Transcript file extension to match.
        recursive: Whether to search subdirectories.

    Returns:
        A `Dataset` with `audio` and `transcription` columns.

    Raises:
        FileNotFoundError: If no audio files, or no complete pairs, exist.
    """
    if recursive:
        pattern = os.path.join(directory, "**", f"*{audio_ext}")
        audio_paths = sorted(glob.glob(pattern, recursive=True))
    else:
        pattern = os.path.join(directory, f"*{audio_ext}")
        audio_paths = sorted(glob.glob(pattern))

    if not audio_paths:
        raise FileNotFoundError(
            f"No '*{audio_ext}' files found in {directory} "
            f"(recursive={recursive})."
        )

    audio_files: list[str] = []
    transcriptions: list[str] = []
    missing_transcripts: list[str] = []

    for audio_path in audio_paths:
        transcript_path = os.path.splitext(audio_path)[0] + transcript_ext
        if not os.path.isfile(transcript_path):
            missing_transcripts.append(audio_path)
            continue
        with open(transcript_path, encoding="utf-8") as transcript_file:
            transcription = transcript_file.read().strip()
        audio_files.append(audio_path)
        transcriptions.append(transcription)

    if missing_transcripts:
        preview = ", ".join(
            os.path.basename(path) for path in missing_transcripts[:3]
        )
        print(
            f"Warning: skipped {len(missing_transcripts)} audio file(s) in "
            f"{directory} with no matching '{transcript_ext}' (e.g. {preview})",
            file=sys.stderr,
        )

    if not audio_files:
        raise FileNotFoundError(
            f"Found {len(audio_paths)} '*{audio_ext}' file(s) in {directory} "
            f"but none had a matching '*{transcript_ext}' transcript."
        )

    dataset = Dataset.from_dict(
        {"audio": audio_files, "transcription": transcriptions}
    )
    dataset = dataset.cast_column("audio", Audio())
    print(
        f"Loaded {len(audio_files)} audio/transcript pair(s) from {directory}",
        file=sys.stderr,
    )
    return dataset


SOURCE_TYPES.register(PairedDataset)
