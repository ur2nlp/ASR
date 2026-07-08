"""Dataset dispatcher, preparation, and CTC data collator.

Provides a type-dispatching loader (paired, audiofolder, huggingface, concat)
that produces standardized {audio, transcription} datasets, a two-stage
preparation pipeline (text normalization → feature extraction + label
encoding), runtime loading of external held-out eval sets, and a data collator
that handles the different padding semantics of CTC inputs vs labels.
"""

import glob
import os
import sys
from dataclasses import dataclass

import torch
from datasets import Audio, Dataset, DatasetDict, concatenate_datasets, load_dataset
from omegaconf import DictConfig
from transformers import ProcessorMixin

from src.preprocessing import build_text_normalizer
from src.processors import get_model_spec


# ---------------------------------------------------------------------------
# Dataset dispatcher
# ---------------------------------------------------------------------------

def load_dataset_from_config(
    dataset_config: DictConfig,
    cache_dir: str,
) -> DatasetDict:
    """Load a dataset based on the config's `type` field.

    All loaders produce a DatasetDict with standardized columns:
    {audio: Audio(...), transcription: str}.

    Args:
        dataset_config: The `dataset` section of the Hydra config.
        cache_dir: Directory for caching the untokenized dataset.

    Returns:
        A HuggingFace DatasetDict with at least a 'train' split.
    """
    untokenized_path = os.path.join(cache_dir, "untokenized")
    if os.path.exists(untokenized_path):
        print(f"Loading cached untokenized dataset from {untokenized_path}", file=sys.stderr)
        return DatasetDict.load_from_disk(untokenized_path)

    dataset_type = dataset_config.type
    if dataset_type == "paired":
        dataset = _load_paired(dataset_config)
    elif dataset_type == "audiofolder":
        dataset = load_dataset("audiofolder", data_dir=dataset_config.path)
    elif dataset_type == "huggingface":
        dataset = _load_huggingface(dataset_config)
    elif dataset_type == "concat":
        dataset = _load_concat(dataset_config, cache_dir)
    else:
        raise ValueError(
            f"Unsupported dataset type: {dataset_type}. "
            f"Available: paired, audiofolder, huggingface, concat"
        )

    dataset = _standardize_columns(dataset, dataset_config)

    os.makedirs(untokenized_path, exist_ok=True)
    dataset.save_to_disk(untokenized_path)
    print(f"Saved untokenized dataset to {untokenized_path}", file=sys.stderr)
    return dataset



def _load_paired(config: DictConfig) -> DatasetDict:
    """Load a corpus of matched audio/transcript files.

    Each audio file (e.g. ``utt001.wav``) is paired with a sibling text file of
    the same stem (``utt001.txt``) containing its transcription. This is the
    common layout for field-collected and manually transcribed speech data.

    Two directory layouts are supported:
      - Pre-split: if ``path`` contains any of ``train/``, ``dev/``, ``test/``
        subdirectories, each is loaded as the corresponding split.
      - Flat: otherwise, all pairs directly under ``path`` are loaded into a
        single ``train`` split (a dev split is carved out later via
        ``dev_size``).

    Args:
        config: The `dataset` section of the Hydra config. Reads ``path`` and,
            optionally, ``audio_ext`` (default ``.wav``), ``transcript_ext``
            (default ``.txt``), and ``recursive`` (default ``False``).

    Returns:
        A DatasetDict with an ``audio`` column (cast to the Audio feature) and
        a ``transcription`` column.
    """
    root = config.path
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"Paired dataset path does not exist or is not a directory: {root}"
        )

    audio_ext = config.get("audio_ext", ".wav")
    transcript_ext = config.get("transcript_ext", ".txt")
    recursive = config.get("recursive", False)

    split_subdirs = [
        name for name in ("train", "dev", "test")
        if os.path.isdir(os.path.join(root, name))
    ]

    if split_subdirs:
        splits = {
            name: _build_paired_split(
                os.path.join(root, name), audio_ext, transcript_ext, recursive
            )
            for name in split_subdirs
        }
        return DatasetDict(splits)

    train_split = _build_paired_split(root, audio_ext, transcript_ext, recursive)
    return DatasetDict({"train": train_split})


def _build_paired_split(
    directory: str,
    audio_ext: str,
    transcript_ext: str,
    recursive: bool,
) -> Dataset:
    """Build a single Dataset split from a directory of audio/transcript pairs.

    Args:
        directory: Directory to search for audio files.
        audio_ext: Audio file extension to match (e.g. ``.wav``).
        transcript_ext: Transcript file extension (e.g. ``.txt``).
        recursive: Whether to search subdirectories recursively.

    Returns:
        A Dataset with ``audio`` (cast to the Audio feature) and
        ``transcription`` columns.

    Raises:
        FileNotFoundError: If no audio files or no complete pairs are found.
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
        preview = ", ".join(os.path.basename(path) for path in missing_transcripts[:3])
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


def _load_huggingface(config: DictConfig) -> DatasetDict:
    """Load a dataset from the HuggingFace Hub.

    Note on naming: load_dataset()'s `path` argument is the Hub dataset
    identifier (e.g. "mozilla-foundation/common_voice_16_1"), and its `name`
    argument is the subset/config (e.g. "zu"). We call these `name` and
    `config` in our YAML to avoid the confusion of calling a dataset identifier
    a "path".
    """
    kwargs = {"path": config.name}
    if hasattr(config, "config") and config.config is not None:
        # load_dataset calls this "name"; it is the dataset subset/config
        kwargs["name"] = config.config
    if hasattr(config, "split") and config.split is not None:
        kwargs["split"] = config.split
    if hasattr(config, "trust_remote_code"):
        kwargs["trust_remote_code"] = config.trust_remote_code

    result = load_dataset(**kwargs)
    if isinstance(result, Dataset):
        return DatasetDict({"train": result})
    return result


def _load_concat(config: DictConfig, parent_cache_dir: str) -> DatasetDict:
    """Concatenate multiple dataset sources via recursive dispatch."""
    sources = config.sources
    splits: dict[str, list] = {}

    for idx, source_config in enumerate(sources):
        source_config = DictConfig(source_config)
        source_id = source_config.get("id", f"source_{idx}")
        source_cache = os.path.join(parent_cache_dir, source_id)

        source_dataset = load_dataset_from_config(source_config, source_cache)
        for split_name, split_data in source_dataset.items():
            splits.setdefault(split_name, []).append(split_data)

    return DatasetDict({
        split_name: concatenate_datasets(split_list)
        for split_name, split_list in splits.items()
    })


def _standardize_columns(dataset: DatasetDict, config: DictConfig) -> DatasetDict:
    """Rename columns to {audio, transcription} and cast audio."""
    audio_col = config.get("audio_column", "audio")
    text_col = config.get("text_column", "transcription")

    for split_name in dataset:
        split = dataset[split_name]
        columns = split.column_names

        if audio_col != "audio" and audio_col in columns:
            dataset[split_name] = split.rename_column(audio_col, "audio")
        if text_col != "transcription" and text_col in columns:
            dataset[split_name] = dataset[split_name].rename_column(text_col, "transcription")

    return dataset


# ---------------------------------------------------------------------------
# Train/dev splitting
# ---------------------------------------------------------------------------

def ensure_train_dev_split(
    dataset: DatasetDict,
    dev_size: float | int,
    seed: int = 1,
) -> DatasetDict:
    """Split training data into train/dev if no dev split exists.

    Args:
        dataset: DatasetDict, possibly with only a 'train' split.
        dev_size: Fraction (0-1) or absolute number of dev examples.
        seed: Random seed for reproducibility.

    Returns:
        DatasetDict with at least 'train' and 'dev' splits.
    """
    has_dev = any(
        key not in ("train", "test") for key in dataset.keys()
    )
    if has_dev:
        return dataset

    if dev_size < 1:
        test_size = dev_size
    else:
        test_size = int(dev_size)

    splits = dataset["train"].train_test_split(test_size=test_size, seed=seed)
    result = DatasetDict({"train": splits["train"], "dev": splits["test"]})
    if "test" in dataset:
        result["test"] = dataset["test"]
    return result


# ---------------------------------------------------------------------------
# Two-stage preparation
# ---------------------------------------------------------------------------

def normalize_dataset(
    dataset: DatasetDict,
    preprocessing_config: DictConfig,
) -> DatasetDict:
    """Stage 1: Apply text normalization to transcriptions."""
    normalizer = build_text_normalizer(preprocessing_config)

    for split_name in dataset:
        dataset[split_name] = dataset[split_name].map(
            lambda example: {"transcription": normalizer(example["transcription"])}
        )

    return dataset


def prepare_dataset_for_training(
    dataset: DatasetDict,
    processor: ProcessorMixin,
    model_type: str,
    sampling_rate: int = 16000,
    max_audio_length_seconds: float | None = None,
    cache_dir: str | None = None,
) -> DatasetDict:
    """Stage 2: Feature extraction + label encoding.

    Pre-extracts and caches features so they are not recomputed each epoch.
    This assumes the CNN feature extractor is frozen — an unfrozen extractor
    would produce stale features after the first weight update. The entry point
    enforces this constraint at startup.

    Args:
        dataset: Normalized DatasetDict with {audio, transcription} columns.
        processor: The combined processor (feature extractor + tokenizer).
        model_type: Model type key for looking up input_column.
        sampling_rate: Target audio sampling rate.
        max_audio_length_seconds: Filter out audio longer than this.
        cache_dir: If provided, cache processed dataset to this path.

    Returns:
        DatasetDict ready for training with model-appropriate input columns.
    """
    processed_path = os.path.join(cache_dir, "processed") if cache_dir else None
    if processed_path and os.path.exists(processed_path):
        print(f"Loading cached processed dataset from {processed_path}", file=sys.stderr)
        return DatasetDict.load_from_disk(processed_path)

    spec = get_model_spec(model_type)
    input_column = spec.input_column

    for split_name in dataset:
        split = _cast_and_filter_audio(
            dataset[split_name],
            sampling_rate=sampling_rate,
            max_audio_length_seconds=max_audio_length_seconds,
            split_name=split_name,
        )
        dataset[split_name] = _feature_extract_and_encode(
            split,
            processor=processor,
            input_column=input_column,
            sampling_rate=sampling_rate,
        )

    if processed_path:
        os.makedirs(processed_path, exist_ok=True)
        dataset.save_to_disk(processed_path)
        print(f"Saved processed dataset to {processed_path}", file=sys.stderr)

    return dataset


def _cast_and_filter_audio(
    dataset: Dataset,
    sampling_rate: int,
    max_audio_length_seconds: float | None,
    split_name: str = "",
) -> Dataset:
    """Resample audio to the target rate and optionally drop over-long clips."""
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))

    if max_audio_length_seconds is not None:
        max_samples = int(max_audio_length_seconds * sampling_rate)
        before = len(dataset)
        dataset = dataset.filter(
            lambda example: len(example["audio"]["array"]) <= max_samples
        )
        after = len(dataset)
        if before != after:
            label = f"{split_name}: " if split_name else ""
            print(
                f"Filtered {label}{before} → {after} "
                f"(removed {before - after} examples > {max_audio_length_seconds}s)",
                file=sys.stderr,
            )
    return dataset


def _feature_extract_and_encode(
    dataset: Dataset,
    processor: ProcessorMixin,
    input_column: str,
    sampling_rate: int,
) -> Dataset:
    """Extract input features from audio and encode transcriptions as labels."""

    def _process_example(example):
        audio_array = example["audio"]["array"]

        # extract features
        inputs = processor.feature_extractor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors=None,
        )
        example[input_column] = inputs[input_column][0]

        # encode labels
        example["labels"] = processor.tokenizer(
            example["transcription"],
        ).input_ids

        return example

    return dataset.map(
        _process_example,
        remove_columns=["audio", "transcription"],
    )


def load_external_eval_sets(
    external_eval_sets: list,
    preprocessing_config: DictConfig,
    processor: ProcessorMixin,
    model_type: str,
    sampling_rate: int = 16000,
    max_audio_length_seconds: float | None = None,
) -> dict[str, Dataset]:
    """Load extra held-out evaluation sets for monitoring during training.

    Each entry is a directory of matched audio/transcript files (the same
    ``paired`` layout as the training data), producing an additional eval set
    that the Trainer reports as ``eval_{name}_wer`` / ``eval_{name}_cer``.
    Unlike the training data these sets are processed fresh each run and never
    cached — they are small and typically change less often than the pipeline.

    Args:
        external_eval_sets: List of configs, each with ``name`` and ``path`` and
            optional ``audio_ext`` (default ``.wav``), ``transcript_ext``
            (default ``.txt``), ``recursive`` (default ``False``), and ``type``
            (default ``paired``; the only type currently supported).
        preprocessing_config: The `preprocessing` config, applied to
            transcriptions so references match the training normalization.
        processor: The combined processor (feature extractor + tokenizer) built
            from the training vocab.
        model_type: Model type key, for the input-column lookup.
        sampling_rate: Target audio sampling rate.
        max_audio_length_seconds: Optional filter for over-long clips.

    Returns:
        Mapping from eval-set name to its processed Dataset.

    Raises:
        ValueError: For an unsupported ``type`` or a duplicate set name.
    """
    normalizer = build_text_normalizer(preprocessing_config)
    spec = get_model_spec(model_type)
    input_column = spec.input_column

    eval_sets: dict[str, Dataset] = {}
    for entry in external_eval_sets:
        entry = DictConfig(entry)
        name = entry.name
        if name in eval_sets:
            raise ValueError(f"Duplicate external eval set name: {name}")

        set_type = entry.get("type", "paired")
        if set_type != "paired":
            raise ValueError(
                f"External eval set '{name}' has unsupported type '{set_type}'. "
                f"Only 'paired' (a directory of matched audio/transcript files) "
                f"is supported."
            )

        dataset = _build_paired_split(
            entry.path,
            entry.get("audio_ext", ".wav"),
            entry.get("transcript_ext", ".txt"),
            entry.get("recursive", False),
        )
        dataset = dataset.map(
            lambda example: {"transcription": normalizer(example["transcription"])}
        )
        dataset = _cast_and_filter_audio(
            dataset,
            sampling_rate=sampling_rate,
            max_audio_length_seconds=max_audio_length_seconds,
            split_name=name,
        )
        dataset = _feature_extract_and_encode(
            dataset,
            processor=processor,
            input_column=input_column,
            sampling_rate=sampling_rate,
        )

        eval_sets[name] = dataset
        print(
            f"Loaded external eval set '{name}': {len(dataset)} examples",
            file=sys.stderr,
        )

    return eval_sets


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------

@dataclass
class DataCollatorCTCWithPadding:
    """Pad CTC inputs and labels with appropriate padding semantics.

    Inputs are padded with the feature extractor's padding value; labels are
    padded with -100 (ignored by CTC loss).

    Attributes:
        processor: The combined processor for padding.
        input_column: The name of the input feature column.
        padding: Padding strategy passed to the processor.
    """

    processor: ProcessorMixin
    input_column: str = "input_values"
    padding: bool | str = True

    def __call__(
        self,
        features: list[dict[str, list[int] | torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        # separate inputs and labels (different padding semantics)
        input_features = [
            {self.input_column: feature[self.input_column]} for feature in features
        ]
        # tokenizer.pad() expects dicts with key "input_ids", so we rename
        # from "labels" here and rename back after padding
        label_features = [
            {"input_ids": feature["labels"]} for feature in features
        ]

        # pad inputs
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # pad labels
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # replace padding tokens in labels with -100
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels

        return batch
