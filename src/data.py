"""Dataset dispatcher, preparation, and CTC data collator.

Provides a type-dispatching loader (audiofolder, huggingface, concat) that
produces standardized {audio, transcription} datasets, a two-stage preparation
pipeline (text normalization → feature extraction + label encoding), and a
data collator that handles the different padding semantics of CTC inputs vs
labels.
"""

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
    if dataset_type == "audiofolder":
        dataset = load_dataset("audiofolder", data_dir=dataset_config.path)
    elif dataset_type == "huggingface":
        dataset = _load_huggingface(dataset_config)
    elif dataset_type == "concat":
        dataset = _load_concat(dataset_config, cache_dir)
    else:
        raise ValueError(
            f"Unsupported dataset type: {dataset_type}. "
            f"Available: audiofolder, huggingface, concat"
        )

    dataset = _standardize_columns(dataset, dataset_config)

    os.makedirs(untokenized_path, exist_ok=True)
    dataset.save_to_disk(untokenized_path)
    print(f"Saved untokenized dataset to {untokenized_path}", file=sys.stderr)
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

    # cast audio to target sampling rate
    for split_name in dataset:
        dataset[split_name] = dataset[split_name].cast_column(
            "audio", Audio(sampling_rate=sampling_rate)
        )

    # optionally filter long audio
    if max_audio_length_seconds is not None:
        max_samples = int(max_audio_length_seconds * sampling_rate)
        for split_name in dataset:
            before = len(dataset[split_name])
            dataset[split_name] = dataset[split_name].filter(
                lambda example: len(example["audio"]["array"]) <= max_samples
            )
            after = len(dataset[split_name])
            if before != after:
                print(
                    f"Filtered {split_name}: {before} → {after} "
                    f"(removed {before - after} examples > {max_audio_length_seconds}s)",
                    file=sys.stderr,
                )

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

    for split_name in dataset:
        dataset[split_name] = dataset[split_name].map(
            _process_example,
            remove_columns=["audio", "transcription"],
        )

    if processed_path:
        os.makedirs(processed_path, exist_ok=True)
        dataset.save_to_disk(processed_path)
        print(f"Saved processed dataset to {processed_path}", file=sys.stderr)

    return dataset


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
