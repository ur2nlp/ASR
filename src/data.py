"""Dataset loading and the two-stage preparation pipeline.

The per-type loaders live in `src/sources/`, one `DatasetArtifact` subclass per
dataset type; this module keeps the boundary that resolves one
(`load_dataset_from_config`), the two-stage preparation pipeline (text
normalization → feature extraction + label encoding), and runtime loading of
external held-out eval sets.

The data collators live in `src/collators.py`; they are re-exported here for
callers that imported them from this module.
"""

import os
import sys

from datasets import Audio, Dataset, DatasetDict
from omegaconf import DictConfig
from transformers import ProcessorMixin

from src.collators import (
    DataCollatorCTCWithPadding,
    DataCollatorSpeechSeq2SeqWithPadding,
)
from src.preprocessing import build_text_normalizer
from src.processors import get_model_spec
from src.sources import make_source
from src.sources.paired import build_paired_split

__all__ = [
    "DataCollatorCTCWithPadding",
    "DataCollatorSpeechSeq2SeqWithPadding",
    "ensure_train_dev_split",
    "load_dataset_from_config",
    "load_external_eval_sets",
    "normalize_dataset",
    "prepare_dataset_for_training",
]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset_from_config(
    dataset_config: DictConfig,
    cache_dir: str,
    seed: int = 1,
    fresh: bool = False,
) -> DatasetDict:
    """Load the dataset a config describes, from cache when one is valid.

    The loaders themselves are `DatasetArtifact` subclasses in `src/sources/`,
    one per `type`, each declaring the parameters its cache is keyed on. This
    function is the boundary: it turns a config into an artifact and resolves
    it. The cache check, the config record and the write are the artifact
    layer's, not this module's.

    The record lives at `<cache_dir>/untokenized/config.yaml`, inside the
    directory it describes. Caches written before the port keep their record
    *beside* that directory, as `<cache_dir>/dataset_config.yaml`, and are
    refused until `tools/migrate_dataset_records.py` has moved them.

    Args:
        dataset_config: The `dataset` section of the Hydra config.
        cache_dir: Directory the `untokenized` subdirectory is created in.
        seed: Global random seed, passed to sources that sample.
        fresh: Rebuild even if a valid cache exists.

    Returns:
        A `DatasetDict` with standardized `{audio, transcription}` columns and
        at least a `train` split.

    Raises:
        ValueError: If no source type is registered under the config's `type`.
        MissingConfigRecordError: If a cached dataset carries no config record.
        ConfigMismatchError: If a cached dataset was built with other settings.
    """
    return make_source(cache_dir, dataset_config, seed).resolve(fresh=fresh)


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
    max_label_length: int | None = None,
    processed_path: str | None = None,
) -> DatasetDict:
    """Stage 2: Feature extraction + label encoding.

    Pre-extracts and caches features so they are not recomputed each epoch.
    For models whose front end learns from raw audio (wav2vec2, HuBERT,
    W2V-BERT) this assumes that front end is frozen — otherwise the cached
    features go stale after the first weight update, and the entry point
    enforces the constraint at startup. Models with a deterministic front end
    (Whisper's mel spectrogram) are unaffected.

    Args:
        dataset: Normalized DatasetDict with {audio, transcription} columns.
        processor: The combined processor (feature extractor + tokenizer).
        model_type: Model type key for looking up input_column.
        sampling_rate: Target audio sampling rate.
        max_audio_length_seconds: Filter out audio longer than this.
        max_label_length: Filter out examples whose encoded labels exceed this
            many tokens. Needed for seq2seq decoders with a hard positional
            limit (Whisper: 448).
        processed_path: If provided, the directory to cache the processed
            dataset in. Callers pass a per-configuration subcache (see
            `artifact_configs.processed_cache_dirname`) so that caches for
            different models coexist instead of invalidating each other.

    Returns:
        DatasetDict ready for training with model-appropriate input columns.
    """
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
        split = _feature_extract_and_encode(
            split,
            processor=processor,
            input_column=input_column,
            sampling_rate=sampling_rate,
        )
        dataset[split_name] = _filter_long_labels(
            split,
            max_label_length=max_label_length,
            split_name=split_name,
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

        # record length for length-grouped batching (group_by_length). This is
        # measured on the raw audio rather than the extracted features so it
        # stays meaningful across architectures: feature length is proportional
        # to it for wav2vec2/W2V-BERT, and constant for models that pad to a
        # fixed window (Whisper), where the feature length would sort nothing.
        example["input_length"] = len(audio_array)

        # encode labels
        example["labels"] = processor.tokenizer(
            example["transcription"],
        ).input_ids
        example["label_length"] = len(example["labels"])

        return example

    return dataset.map(
        _process_example,
        remove_columns=["audio", "transcription"],
    )


def _filter_long_labels(
    dataset: Dataset,
    max_label_length: int | None,
    split_name: str = "",
) -> Dataset:
    """Drop examples whose encoded labels exceed the decoder's position limit.

    Seq2seq decoders have a hard maximum target length (448 for Whisper), and
    an over-long label raises an index error deep in the forward pass rather
    than a readable message, so we filter up front.
    """
    if max_label_length is None:
        return dataset

    before = len(dataset)
    dataset = dataset.filter(
        lambda example: example["label_length"] <= max_label_length
    )
    after = len(dataset)
    if before != after:
        label = f"{split_name}: " if split_name else ""
        print(
            f"Filtered {label}{before} → {after} "
            f"(removed {before - after} examples with > {max_label_length} label tokens)",
            file=sys.stderr,
        )
    return dataset


def load_external_eval_sets(
    external_eval_sets: list,
    preprocessing_config: DictConfig,
    processor: ProcessorMixin,
    model_type: str,
    sampling_rate: int = 16000,
    max_audio_length_seconds: float | None = None,
    max_label_length: int | None = None,
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
        max_label_length: Optional filter for over-long encoded transcripts.

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

        dataset = build_paired_split(
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
        dataset = _filter_long_labels(
            dataset,
            max_label_length=max_label_length,
            split_name=name,
        )

        eval_sets[name] = dataset
        print(
            f"Loaded external eval set '{name}': {len(dataset)} examples",
            file=sys.stderr,
        )

    return eval_sets
