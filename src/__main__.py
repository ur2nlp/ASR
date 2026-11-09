"""Hydra entry point for ASR fine-tuning.

Orchestrates the full pipeline: cache management → shared feature cache →
split assignment → text normalization → vocab generation → processor
construction → label encoding → model setup → training with callbacks →
held-out test evaluation.
"""

import os
import shutil
import sys

import hydra
from omegaconf import DictConfig, OmegaConf
from transformers.trainer_utils import get_last_checkpoint

from src.artifact_configs import FeatureCacheConfig, warn_on_legacy_processed_cache
from src import focus, splits
from src.callbacks import (
    DelayedEarlyStoppingCallback,
    DetectBrokenLossCallback,
    UnfreezeCallback,
)
from src.data import (
    AUDIO_PATH_COLUMN,
    SOURCE_SPLIT_COLUMN,
    assemble_splits,
    build_feature_cache,
    default_assignment,
    encode_labels,
    load_dataset_from_config,
    load_external_eval_sets,
    load_feature_cache,
    normalize_texts,
)
from src.metrics import make_compute_metrics, preprocess_logits_for_metrics
from src.models import setup_model
from src.processors import (
    ModelSpec,
    build_feature_extractor,
    get_model_spec,
    setup_processor,
    setup_tokenizer,
)


OmegaConf.register_new_resolver("divide", lambda x, y: int(x / y), replace=True)


def _build_output_dir(args: DictConfig) -> str:
    """Build the output directory for this run.

    The path encodes the experiment conditions as
    ``{output_dir}/{id}/{model_short}_{training}``, followed by the FOCUS
    vocabulary and the metadata split id when those are enabled. If
    ``experiment_id`` is set,
    it is appended as a suffix so that repeated runs over the same model and
    training config (e.g. a hyperparameter sweep) do not overwrite each other
    while keeping the descriptive path.
    """
    language = args.dataset.id
    model_short = args.model.get("short_name", args.model.type)
    training_name = args.training.name
    base_path = os.path.join(args.output_dir, language, f"{model_short}_{training_name}")

    # A replaced vocabulary is a different experiment, not a rerun of the same
    # one, so it must not land on the baseline's checkpoints. `vocabulary_id`
    # rather than `tokenizer_id`: the latter ends with the model slug, which
    # `model_short` has already put in this path.
    if focus.is_enabled(args):
        base_path = f"{base_path}_{focus.vocabulary_id(args)}"

    # Each fold of a cross-validation is its own experiment; without this they
    # would all write to one directory.
    if splits.is_enabled(args):
        base_path = f"{base_path}_{splits.split_id(args)}"

    experiment_id = args.get("experiment_id")
    if experiment_id:
        return f"{base_path}_{experiment_id}"
    return base_path


def _handle_cache_cleanup(
    args: DictConfig,
    cache_dir: str,
    output_dir: str,
    features_path: str,
) -> None:
    """Conditionally clear cached artifacts based on fresh_* flags.

    `fresh_processed` clears only this run's feature cache, leaving caches for
    other feature extractors intact. Note that the cache is shared by every
    model with the same extractor (and every split), so it is rebuilt for all of
    them. `fresh_dataset` still clears everything, split directories included.
    """
    if args.get("fresh_model", False):
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            print(f"Cleared model output dir: {output_dir}", file=sys.stderr)

    if args.get("fresh_processed", False):
        if os.path.exists(features_path):
            shutil.rmtree(features_path)
            print(f"Cleared feature cache: {features_path}", file=sys.stderr)

    if args.get("fresh_dataset", False):
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print(f"Cleared all dataset caches: {cache_dir}", file=sys.stderr)


def _build_training_arguments(
    args: DictConfig,
    spec: ModelSpec,
    output_dir: str,
    metric_for_best_model: str | None = None,
):
    """Construct TrainingArguments from config.

    The class is taken from the ModelSpec so that generation-based evaluation
    gets a ``Seq2SeqTrainingArguments`` carrying the extra generation knobs.

    Args:
        args: Full Hydra config.
        spec: The ModelSpec for the configured architecture.
        output_dir: Directory for checkpoints and the best model.
        metric_for_best_model: Overrides ``training.metric_for_best_model`` when
            given. Used when external eval sets turn ``eval_dataset`` into a dict,
            which makes HuggingFace prefix each metric with its set name (so the
            dev metric becomes e.g. ``dev_wer`` rather than ``wer``).
    """
    training = args.training
    if metric_for_best_model is None:
        metric_for_best_model = training.metric_for_best_model

    extra_kwargs = {}
    if spec.uses_generation:
        # evaluation must decode autoregressively rather than score logits,
        # otherwise WER would be computed over teacher-forced predictions
        extra_kwargs["predict_with_generate"] = True
        extra_kwargs["generation_max_length"] = training.get(
            "generation_max_length", 225
        )
        extra_kwargs["generation_num_beams"] = training.get("generation_num_beams", 1)

    return spec.training_arguments_class(
        output_dir=output_dir,
        num_train_epochs=training.num_train_epochs,
        per_device_train_batch_size=training.per_device_train_batch_size,
        per_device_eval_batch_size=training.per_device_eval_batch_size,
        gradient_accumulation_steps=training.gradient_accumulation_steps,
        learning_rate=training.learning_rate,
        lr_scheduler_type=training.lr_scheduler_type,
        warmup_ratio=training.warmup_ratio,
        weight_decay=training.weight_decay,
        max_grad_norm=training.max_grad_norm,
        max_steps=training.get("max_steps", -1),
        eval_strategy=training.eval_strategy,
        eval_steps=training.get("eval_steps"),
        save_strategy=training.save_strategy,
        save_steps=training.get("save_steps"),
        save_total_limit=training.save_total_limit,
        load_best_model_at_end=training.load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=training.greater_is_better,
        logging_steps=training.logging_steps,
        bf16=training.get("bf16", False),
        fp16=training.get("fp16", False),
        optim=training.get("optim", "adamw_torch"),
        dataloader_num_workers=training.get("dataloader_num_workers", 0),
        gradient_checkpointing=training.get("gradient_checkpointing", False),
        # Reentrant checkpointing re-runs each checkpointed block inside
        # backward and frees the saved graph as it goes, which an
        # encoder-decoder trips over: Whisper's decoder attends to encoder
        # states whose graph the encoder's own backward has already released,
        # and the step dies with "Trying to backward through the graph a second
        # time". The non-reentrant implementation keeps them alive.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=training.get("group_by_length", False),
        length_column_name="input_length",
        seed=args.seed,
        report_to="none",
        remove_unused_columns=False,
        **extra_kwargs,
    )


def _build_data_collator(spec: ModelSpec, processor, model):
    """Instantiate the architecture's data collator.

    Seq2seq collators additionally need the model's decoder start token so they
    can strip it from the labels; the CTC collator has no such argument.
    """
    collator_kwargs = {
        "processor": processor,
        "input_column": spec.input_column,
    }
    if spec.objective == "seq2seq":
        collator_kwargs["decoder_start_token_id"] = model.config.decoder_start_token_id

    return spec.collator_class(**collator_kwargs)


def _build_callbacks(args: DictConfig) -> list:
    """Construct trainer callbacks from config."""
    callbacks = [DetectBrokenLossCallback()]

    training = args.training

    # early stopping
    patience = training.get("early_stopping_patience")
    if patience is not None:
        delay_ratio = training.get("early_stopping_delay_ratio", 0.0)
        delay_steps = int(delay_ratio * training.get("max_steps", 0))
        callbacks.append(
            DelayedEarlyStoppingCallback(
                early_stopping_patience=patience,
                early_stopping_delay_steps=delay_steps,
            )
        )

    # unfreeze scheduling
    unfreeze_ratio = training.get("unfreeze_step_ratio")
    if unfreeze_ratio is not None:
        unfreeze_step = int(unfreeze_ratio * training.max_steps)
        callbacks.append(UnfreezeCallback(unfreeze_step=unfreeze_step))

    return callbacks


def _resolve_external_eval_sets(args: DictConfig) -> list | None:
    """Resolve external eval set configs from the config group or a CLI override.

    Supports both ``external_eval=<name>`` (config group) and a direct
    ``external_eval_sets=[...]`` override on the command line.
    """
    external_eval_sets = args.get("external_eval_sets")
    if external_eval_sets is None and "external_eval" in args:
        external_eval_sets = args.external_eval.get("external_eval_sets")
    return external_eval_sets


def _add_external_eval_sets(
    eval_dataset,
    primary_key: str | None,
    external_eval_sets: list,
    metric_for_best_model: str,
    args: DictConfig,
    processor,
) -> tuple[dict, str]:
    """Fold external eval sets into a dict-valued eval_dataset.

    When ``eval_dataset`` becomes a dict, HuggingFace prefixes each metric with
    its set name, so best-checkpoint selection must track the primary dev set's
    qualified metric (e.g. ``dev_wer``). Returns the eval dict and the possibly
    rewritten ``metric_for_best_model``.
    """
    eval_dict: dict = {}
    if eval_dataset is not None:
        primary_key = primary_key or "dev"
        eval_dict[primary_key] = eval_dataset
    else:
        primary_key = None
        print(
            "Warning: external eval sets configured but no dev/test split exists; "
            "best-checkpoint selection may not track a meaningful metric.",
            file=sys.stderr,
        )

    extra_sets = load_external_eval_sets(
        external_eval_sets,
        preprocessing_config=args.preprocessing,
        processor=processor,
        model_type=args.model.type,
        sampling_rate=args.audio.sampling_rate,
        max_audio_length_seconds=args.dataset.get("max_audio_length_seconds"),
        max_label_length=args.dataset.get("max_label_length"),
    )
    for name, dataset in extra_sets.items():
        if name in eval_dict:
            raise ValueError(
                f"External eval set name '{name}' conflicts with an existing "
                f"eval set: {list(eval_dict.keys())}"
            )
        eval_dict[name] = dataset

    if primary_key is not None:
        already_qualified = metric_for_best_model.startswith(
            (f"{primary_key}_", f"eval_{primary_key}_")
        )
        if not already_qualified:
            metric_for_best_model = f"{primary_key}_{metric_for_best_model}"

    return eval_dict, metric_for_best_model


def _load_or_build_features(
    args: DictConfig,
    feature_config: FeatureCacheConfig,
    feature_extractor,
    features_path: str,
):
    """Return the shared feature cache, extracting it first if needed.

    The untokenized source is only resolved when the features have to be
    built: once they exist, a run never touches the audio again.

    A directory without its `config.yaml` is a build that did not finish; the
    record is written last, so it is discarded and rebuilt.

    Returns:
        The loaded features.
    """
    config_path = os.path.join(features_path, "config.yaml")

    if os.path.exists(config_path):
        feature_config.source_splits = list(OmegaConf.load(config_path).source_splits)
        feature_config.check_cached(config_path)
        print(f"Loading cached features from {features_path}", file=sys.stderr)
        return load_feature_cache(features_path, feature_config.source_splits)

    if os.path.isdir(features_path):
        print(f"Discarding incomplete feature cache at {features_path}", file=sys.stderr)
        shutil.rmtree(features_path)

    # the cache check, config record and write of the untokenized source belong
    # to the artifact layer: each source in `src/sources/` declares what its
    # cache is keyed on, and `resolve()` does the rest
    source = load_dataset_from_config(args.dataset, args.dataset.cache_dir, seed=args.seed)
    feature_config.source_splits = list(source.keys())
    build_feature_cache(
        source,
        feature_extractor,
        input_column=feature_config.input_column,
        sampling_rate=feature_config.sampling_rate,
        cache_path=features_path,
    )
    feature_config.save(config_path)
    return load_feature_cache(features_path, feature_config.source_splits)


def _assign_splits(args: DictConfig, features, cache_dir: str) -> tuple[str, dict]:
    """Assign every example to train/dev/test.

    Without a metadata split the source's own splits are used, with dev
    sampled from train if the source has none; split-dependent artifacts (the
    CTC vocabulary, a FOCUS tokenizer) then live directly under `cache_dir` as
    they always have. With one, they live under `cache_dir/splits/<split_id>/`,
    since each split configuration trains on different rows.

    Returns:
        A tuple of the directory for split-dependent artifacts and
        `{split: example indices}`.
    """
    source_splits = features[SOURCE_SPLIT_COLUMN]
    if not splits.is_enabled(args):
        return cache_dir, default_assignment(source_splits, args.dataset.dev_size, seed=args.seed)

    spec = splits.SplitSpec.from_config(args.split)
    distinct_sources = sorted(set(source_splits))
    if len(distinct_sources) > 1:
        print(
            f"Note: the source's own splits {distinct_sources} are pooled and "
            f"reassigned by split.{spec.name}.",
            file=sys.stderr,
        )

    key_column = AUDIO_PATH_COLUMN if spec.key_from == "path" else spec.key_from
    if key_column not in features.column_names:
        raise ValueError(
            f"split.key_from={spec.key_from!r} names a column the dataset does not "
            f"have (columns: {features.column_names})."
        )
    key_sources = features[key_column]
    # `input_length` is the raw audio length in samples at the extraction rate
    durations = [
        length / args.audio.sampling_rate for length in features["input_length"]
    ]
    assignment = splits.assign_splits(
        key_sources,
        spec,
        seed=args.seed,
        durations=durations,
        max_duration=args.dataset.get("max_audio_length_seconds"),
    )

    split_root = os.path.join(cache_dir, "splits", spec.split_id(args.seed))
    splits.save_split_record(
        split_root, spec, args.seed, assignment, key_sources, durations=durations
    )
    return split_root, assignment


def _validate_config(args: DictConfig, spec: ModelSpec) -> None:
    """Reject config combinations that the architecture cannot honour.

    Checks that would otherwise surface as a confusing runtime failure many
    minutes into a run, or worse, as a silently wrong result.
    """
    # Pre-extracted features go stale if a *learned* front end keeps training.
    # Whisper's front end is a deterministic mel spectrogram, so unfreezing its
    # conv stack is safe and does not trip this.
    if spec.learned_feature_extractor:
        if not args.training.get("freeze_feature_extractor", True):
            raise ValueError(
                f"freeze_feature_extractor=false is not supported for model type "
                f"'{args.model.type}'. Feature extraction is pre-computed and cached "
                f"before training, so an unfrozen feature extractor would train on "
                f"stale features. Full end-to-end fine-tuning is not yet implemented "
                f"in this framework."
            )

    # Fail before the dataset is loaded rather than at tokenizer setup, which is
    # several minutes of audio decoding later.
    if focus.is_enabled(args) and spec.tokenizer_source != "pretrained":
        raise ValueError(
            f"focus.enabled=true is not supported for model type "
            f"'{args.model.type}'. FOCUS replaces a pretrained subword "
            f"vocabulary with one learned from the target transcripts, but this "
            f"architecture already builds its vocabulary from them "
            f"(tokenizer_source='{spec.tokenizer_source}'). Set focus=none."
        )

    if args.lm.get("enabled", False) and not spec.supports_lm_decoding:
        raise ValueError(
            f"lm.enabled=true is not supported for model type '{args.model.type}'. "
            f"KenLM shallow fusion via pyctcdecode operates on per-frame CTC logits "
            f"and has no equivalent for an autoregressive decoder. Set lm.enabled=false."
        )

    if args.training.get("group_by_length", False) and spec.fixed_length_features:
        print(
            f"Warning: group_by_length=true has no effect for model type "
            f"'{args.model.type}', whose inputs are all padded to a fixed length. "
            f"Set it to false to skip the sampler's bookkeeping.",
            file=sys.stderr,
        )


@hydra.main(version_base=None, config_path="../configs", config_name="main")
def main(args: DictConfig) -> None:
    """Main training entry point."""
    print(f"Config:\n{OmegaConf.to_yaml(args)}", file=sys.stderr)

    spec = get_model_spec(args.model.type)
    _validate_config(args, spec)

    cache_dir = args.dataset.cache_dir
    output_dir = _build_output_dir(args)

    # Features depend only on the audio and the feature extractor, so one cache
    # serves every model sharing that extractor and every split configuration.
    feature_extractor = build_feature_extractor(args, spec)
    feature_config = FeatureCacheConfig.from_args(args, feature_extractor)
    features_path = os.path.join(cache_dir, feature_config.dirname())

    # cache cleanup
    _handle_cache_cleanup(args, cache_dir, output_dir, features_path)
    warn_on_legacy_processed_cache(cache_dir)

    # --- Stage 1: Shared feature cache ---
    features = _load_or_build_features(args, feature_config, feature_extractor, features_path)

    # --- Stage 2: Assign splits and normalize text ---
    # Everything from here on is per run and by index: nothing rewrites the
    # cached features.
    split_root, assignment = _assign_splits(args, features, cache_dir)
    texts = normalize_texts(features["transcription"], args.preprocessing)

    # --- Stage 3: Build tokenizer and processor ---
    # CTC models generate a character vocab from the training transcripts;
    # seq2seq models reuse the checkpoint's own subword tokenizer, or train a
    # FOCUS one on the training transcripts. Both live under `split_root`, so
    # each split configuration gets its own.
    train_texts = [texts[index] for index in assignment["train"]]
    tokenizer = setup_tokenizer(
        args,
        train_texts=train_texts,
        vocab_dir=os.path.join(split_root, "vocab"),
        cache_dir=split_root,
    )
    processor = setup_processor(args, tokenizer)

    # --- Stage 4: Encode labels and assemble splits ---
    used_indices = sorted(index for indices in assignment.values() for index in indices)
    labels = encode_labels(texts, tokenizer, used_indices)
    dataset = assemble_splits(
        features,
        assignment,
        labels,
        input_column=spec.input_column,
        sampling_rate=args.audio.sampling_rate,
        max_audio_length_seconds=args.dataset.get("max_audio_length_seconds"),
        max_label_length=args.dataset.get("max_label_length"),
    )
    print(f"Dataset splits: {dict(dataset.num_rows)}", file=sys.stderr)

    # --- Stage 5: Setup model ---
    model = setup_model(args, processor, cache_dir=split_root)

    # --- Stage 6: Train ---
    callbacks = _build_callbacks(args)
    compute_metrics = make_compute_metrics(
        processor,
        prediction_decode_kwargs=spec.prediction_decode_kwargs,
        label_decode_kwargs=spec.label_decode_kwargs,
    )

    data_collator = _build_data_collator(spec, processor, model)

    # determine eval dataset (prefer dev, fall back to test)
    eval_split = "dev" if "dev" in dataset else "test" if "test" in dataset else None
    eval_dataset = dataset[eval_split] if eval_split else None
    metric_for_best_model = args.training.metric_for_best_model

    # optional external held-out eval sets (config group or CLI override)
    external_eval_sets = _resolve_external_eval_sets(args)
    if external_eval_sets:
        eval_dataset, metric_for_best_model = _add_external_eval_sets(
            eval_dataset=eval_dataset,
            primary_key=eval_split,
            external_eval_sets=external_eval_sets,
            metric_for_best_model=metric_for_best_model,
            args=args,
            processor=processor,
        )

    training_args = _build_training_arguments(
        args, spec, output_dir, metric_for_best_model=metric_for_best_model
    )

    trainer_kwargs = {}
    if not spec.uses_generation:
        # under predict_with_generate the Trainer already returns token IDs,
        # so there are no logits left to reduce
        trainer_kwargs["preprocess_logits_for_metrics"] = preprocess_logits_for_metrics

    trainer = spec.trainer_class(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        **trainer_kwargs,
    )

    # save full config
    os.makedirs(output_dir, exist_ok=True)
    config_save_path = os.path.join(output_dir, "training_config.yaml")
    with open(config_save_path, "w") as f:
        OmegaConf.save(args, f)

    # train; when preempt_resume is set (preempt + requeue cluster workflows),
    # fall back to the latest checkpoint in output_dir if none is given explicitly
    checkpoint = args.get("resume_from_checkpoint")
    if checkpoint is None and args.get("preempt_resume", False):
        checkpoint = get_last_checkpoint(output_dir)
        if checkpoint is not None:
            print(f"Auto-resuming from checkpoint: {checkpoint}", file=sys.stderr)
    trainer.train(resume_from_checkpoint=checkpoint)

    # save best model
    best_model_dir = os.path.join(output_dir, "best-checkpoint")
    trainer.save_model(best_model_dir)
    processor.save_pretrained(best_model_dir)

    # --- Stage 7: Evaluate the held-out test split (opt-in) ---
    # Off by default: scoring test at the end of every run invites selecting
    # runs on it. Only applies when a `test` split exists and was not already
    # the eval set. `save_metrics` writes `test_results.json` at the run root.
    if "test" in dataset and eval_split != "test" and args.training.get("evaluate_test", False):
        test_metrics = trainer.evaluate(dataset["test"], metric_key_prefix="test")
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)

    # Write trainer_state.json at the run root. `save_model` does not -- it saves
    # the model and tokenizer only -- and the state it holds is the sole record
    # that a run finished: only the copy written here carries the `train_runtime`
    # that the final `train()` log entry adds. `tools/remote_inventory.sh` looks
    # for it first and drops a run entirely when it is missing, and
    # `fetch_diff.get_local_status` needs it to call a run complete rather than
    # still training. Without it a finished run is invisible to the fetcher.
    trainer.save_state()

    # copy config to best checkpoint for reproducibility
    with open(os.path.join(best_model_dir, "training_config.yaml"), "w") as f:
        OmegaConf.save(args, f)

    print(f"Training complete. Best model saved to {best_model_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
