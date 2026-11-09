"""Dataset loading and the two-stage preparation pipeline.

The per-type loaders live in `src/sources/`, one `DatasetArtifact` subclass per
dataset type; this module keeps the boundary that resolves one
(`load_dataset_from_config`), the shared feature cache, the per-run stages
applied over it by index (text normalization, label encoding, split assignment,
length filtering), and runtime loading of external held-out eval sets.

The data collators live in `src/collators.py`; they are re-exported here for
callers that imported them from this module.
"""

import os
import sys

from datasets import Audio, Dataset, DatasetDict, concatenate_datasets
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
    "assemble_splits",
    "build_feature_cache",
    "default_assignment",
    "encode_labels",
    "load_dataset_from_config",
    "load_external_eval_sets",
    "load_feature_cache",
    "normalize_texts",
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
# Feature cache
# ---------------------------------------------------------------------------

AUDIO_PATH_COLUMN = "audio_path"
SOURCE_SPLIT_COLUMN = "source_split"
EXAMPLE_INDEX_COLUMN = "example_index"


def build_feature_cache(
    dataset: DatasetDict,
    feature_extractor,
    input_column: str,
    sampling_rate: int,
    cache_path: str,
) -> None:
    """Extract features for every example of every source split, once.

    Each source split is written to `<cache_path>/<split>.arrow` directly by
    `Dataset.map`, so the features are written once and never copied, and
    nothing lands in the source cache's directory (a `map` without an explicit
    `cache_file_name` writes a full copy of its output beside its input, which
    is how the untokenized cache grew to many times its size).

    No examples are dropped here. The length filters, the text normalization,
    the label encoding and the train/dev/test assignment all vary between runs
    that share these features, so they are applied afterwards by index (see
    `assemble_splits`).

    Besides the features, each row keeps every non-audio source column, the
    audio file's basename (`audio_path`), its source split name, and
    `input_length` -- the raw audio length in samples, which is what the audio
    length filter and length-grouped batching read.

    Args:
        dataset: The untokenized source.
        feature_extractor: The model's feature extractor.
        input_column: Column to store the features under.
        sampling_rate: Rate to resample the audio to.
        cache_path: Directory to write into; created if missing.
    """
    os.makedirs(cache_path, exist_ok=True)

    for split_name in dataset:
        split = dataset[split_name].cast_column("audio", Audio(sampling_rate=sampling_rate))

        def _extract(example, split_name=split_name):
            audio = example["audio"]
            audio_array = audio["array"]
            inputs = feature_extractor(
                audio_array,
                sampling_rate=sampling_rate,
                return_tensors=None,
            )
            return {
                input_column: inputs[input_column][0],
                # measured on the raw audio rather than the features so that it
                # stays meaningful for models padding to a fixed window
                # (Whisper), where the feature length would sort nothing
                "input_length": len(audio_array),
                AUDIO_PATH_COLUMN: os.path.basename(audio["path"]) if audio["path"] else None,
                SOURCE_SPLIT_COLUMN: split_name,
            }

        final_path = os.path.join(cache_path, f"{split_name}.arrow")
        partial_path = final_path + ".partial"
        split.map(
            _extract,
            remove_columns=["audio"],
            cache_file_name=partial_path,
            load_from_cache_file=False,
            desc=f"Extracting features ({split_name})",
        )
        os.replace(partial_path, final_path)
        print(f"Wrote features for {split_name} to {final_path}", file=sys.stderr)


def load_feature_cache(cache_path: str, source_splits: list[str]) -> Dataset:
    """Load a feature cache as one memory-mapped dataset, source splits in order.

    Columns not shared by every source split are dropped, so sources whose
    splits carry different extra columns still concatenate.
    """
    parts = [
        Dataset.from_file(os.path.join(cache_path, f"{split_name}.arrow"))
        for split_name in source_splits
    ]
    if len(parts) == 1:
        return parts[0]

    shared = [
        column for column in parts[0].column_names
        if all(column in part.column_names for part in parts[1:])
    ]
    return concatenate_datasets([part.select_columns(shared) for part in parts])


# ---------------------------------------------------------------------------
# Text and labels
# ---------------------------------------------------------------------------

def normalize_texts(texts: list[str], preprocessing_config: DictConfig) -> list[str]:
    """Apply the configured text normalization to a list of transcripts."""
    normalizer = build_text_normalizer(preprocessing_config)
    return [normalizer(text) for text in texts]


def encode_labels(
    texts: list[str],
    tokenizer,
    indices: list[int],
) -> list[list[int]]:
    """Encode the transcripts at `indices`; every other row gets an empty label.

    Only the rows that are actually used are encoded -- an excluded transcriber
    or a held-out English test set may hold characters the training vocabulary
    lacks, and there is no reason to spend time on rows that are discarded.

    Args:
        texts: Normalized transcripts, one per example.
        tokenizer: The run's tokenizer.
        indices: Rows to encode.

    Returns:
        One label list per example.
    """
    labels: list[list[int]] = [[] for _ in texts]
    for index in indices:
        labels[index] = tokenizer(texts[index]).input_ids
    return labels


# ---------------------------------------------------------------------------
# Split assignment and assembly
# ---------------------------------------------------------------------------

def default_assignment(
    source_splits: list[str],
    dev_size: float | int,
    seed: int = 1,
) -> dict[str, list[int]]:
    """Assign examples to splits from the source's own split names.

    Used when no metadata split is configured. Every source split keeps its
    name. If the source has no split other than `train` and `test`, a dev split
    is sampled from `train` -- by example, exactly as `train_test_split` with
    this seed always has, so a vocabulary cached under the old pipeline still
    corresponds to the same training rows.

    Args:
        source_splits: Each example's source split name, in feature-cache order.
        dev_size: Fraction (0-1) or absolute number of dev examples.
        seed: Random seed for the dev sample.

    Returns:
        `{split: example indices}`.
    """
    assignment: dict[str, list[int]] = {}
    for index, split_name in enumerate(source_splits):
        assignment.setdefault(split_name, []).append(index)

    has_dev = any(name not in ("train", "test") for name in assignment)
    if has_dev or "train" not in assignment:
        return assignment

    if dev_size < 1:
        test_size = dev_size
    else:
        test_size = int(dev_size)

    train_indices = Dataset.from_dict({"index": assignment["train"]})
    halves = train_indices.train_test_split(test_size=test_size, seed=seed)
    assignment["train"] = halves["train"]["index"]
    assignment["dev"] = halves["test"]["index"]
    return assignment


def assemble_splits(
    features: Dataset,
    assignment: dict[str, list[int]],
    labels: list[list[int]],
    input_column: str,
    sampling_rate: int,
    max_audio_length_seconds: float | None = None,
    max_label_length: int | None = None,
) -> DatasetDict:
    """Combine cached features with this run's labels and select each split.

    Nothing here rewrites the features: the labels are attached as an in-memory
    column beside the memory-mapped cache, and each split is an index selection
    over the result.

    Args:
        features: The loaded feature cache.
        assignment: `{split: example indices}`.
        labels: One encoded label list per example.
        input_column: The features column.
        sampling_rate: Rate the features were extracted at, for the audio
            length filter.
        max_audio_length_seconds: Drop examples with longer audio.
        max_label_length: Drop examples whose labels exceed this many tokens.
            Needed for seq2seq decoders with a hard positional limit (Whisper:
            448), which otherwise fail with an index error deep in the forward
            pass.

    Returns:
        A `DatasetDict` with `input_column`, `input_length`, `labels`,
        `label_length` and `example_index` (the row in `features`) per split.
        The collators read only the input and label columns, so the extra
        columns pass through training untouched.
    """
    input_lengths = features["input_length"]
    max_samples = None
    if max_audio_length_seconds is not None:
        max_samples = int(max_audio_length_seconds * sampling_rate)

    label_lengths = [len(label) for label in labels]
    with_labels = concatenate_datasets(
        [
            features.select_columns([input_column, "input_length"]),
            Dataset.from_dict({
                "labels": labels,
                "label_length": label_lengths,
                # row in the feature cache, so a prediction can be traced back
                # to its audio file (and from there to its metadata)
                EXAMPLE_INDEX_COLUMN: list(range(len(features))),
            }),
        ],
        axis=1,
    )

    result = DatasetDict()
    for split_name, indices in assignment.items():
        kept = []
        too_long_audio = 0
        too_long_labels = 0
        for index in indices:
            if max_samples is not None and input_lengths[index] > max_samples:
                too_long_audio += 1
            elif max_label_length is not None and label_lengths[index] > max_label_length:
                too_long_labels += 1
            else:
                kept.append(index)

        if too_long_audio:
            print(
                f"Filtered {split_name}: removed {too_long_audio} examples "
                f"> {max_audio_length_seconds}s",
                file=sys.stderr,
            )
        if too_long_labels:
            print(
                f"Filtered {split_name}: removed {too_long_labels} examples "
                f"with > {max_label_length} label tokens",
                file=sys.stderr,
            )
        result[split_name] = with_labels.select(kept, keep_in_memory=True)

    return result


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
