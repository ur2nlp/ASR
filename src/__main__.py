"""Hydra entry point for ASR fine-tuning.

Orchestrates the full pipeline: cache management → dataset loading →
text normalization → vocab generation → processor construction →
model setup → training with callbacks.
"""

import os
import shutil
import sys

import hydra
from omegaconf import DictConfig, OmegaConf
from transformers.trainer_utils import get_last_checkpoint

from src.artifact_configs import (
    DatasetConfig,
    ProcessedDatasetConfig,
    processed_cache_dirname,
    warn_on_legacy_processed_cache,
)
from src.callbacks import (
    DelayedEarlyStoppingCallback,
    DetectBrokenLossCallback,
    UnfreezeCallback,
)
from src.data import (
    ensure_train_dev_split,
    load_dataset_from_config,
    load_external_eval_sets,
    normalize_dataset,
    prepare_dataset_for_training,
)
from src.metrics import make_compute_metrics, preprocess_logits_for_metrics
from src.models import setup_model
from src.processors import ModelSpec, get_model_spec, setup_processor, setup_tokenizer


OmegaConf.register_new_resolver("divide", lambda x, y: int(x / y), replace=True)


def _build_output_dir(args: DictConfig) -> str:
    """Build the output directory for this run.

    The path encodes the experiment conditions as
    ``{output_dir}/{id}/{model_short}_{training}``. If ``experiment_id`` is set,
    it is appended as a suffix so that repeated runs over the same model and
    training config (e.g. a hyperparameter sweep) do not overwrite each other
    while keeping the descriptive path.
    """
    language = args.dataset.id
    model_short = args.model.get("short_name", args.model.type)
    training_name = args.training.name
    base_path = os.path.join(args.output_dir, language, f"{model_short}_{training_name}")

    experiment_id = args.get("experiment_id")
    if experiment_id:
        return f"{base_path}_{experiment_id}"
    return base_path


def _handle_cache_cleanup(
    args: DictConfig,
    cache_dir: str,
    output_dir: str,
    processed_path: str,
) -> None:
    """Conditionally clear cached artifacts based on fresh_* flags.

    `fresh_processed` clears only this run's own processed subcache, leaving
    other models' subcaches for the same dataset intact — rebuilding a Whisper
    cache should not throw away the wav2vec2 one beside it. `fresh_dataset`
    still clears everything, subcaches included.
    """
    if args.get("fresh_model", False):
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            print(f"Cleared model output dir: {output_dir}", file=sys.stderr)

    if args.get("fresh_processed", False):
        if os.path.exists(processed_path):
            shutil.rmtree(processed_path)
            print(f"Cleared processed dataset cache: {processed_path}", file=sys.stderr)

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

    # feature extraction and label encoding are model-specific, so each
    # configuration caches into its own subdirectory beside the shared
    # untokenized text rather than fighting over a single `processed/`
    processed_path = os.path.join(cache_dir, processed_cache_dirname(args))

    # cache cleanup
    _handle_cache_cleanup(args, cache_dir, output_dir, processed_path)
    warn_on_legacy_processed_cache(cache_dir)

    # --- Stage 1: Load and normalize dataset ---
    dataset_config = DatasetConfig.from_args(args)
    dataset_config_path = os.path.join(cache_dir, "dataset_config.yaml")
    dataset_config.check_cached(dataset_config_path)

    dataset = load_dataset_from_config(args.dataset, cache_dir)
    dataset = normalize_dataset(dataset, args.preprocessing)
    dataset = ensure_train_dev_split(dataset, args.dataset.dev_size, seed=args.seed)

    dataset_config.save(dataset_config_path)
    print(f"Dataset splits: {dict(dataset.num_rows)}", file=sys.stderr)

    # --- Stage 2: Build tokenizer and processor ---
    # CTC models generate a character vocab from the training transcripts;
    # seq2seq models reuse the checkpoint's own subword tokenizer.
    vocab_dir = os.path.join(cache_dir, "vocab")
    tokenizer = setup_tokenizer(
        args,
        train_texts=dataset["train"]["transcription"],
        vocab_dir=vocab_dir,
    )
    processor = setup_processor(args, tokenizer)
    vocab_size = len(tokenizer)

    # --- Stage 3: Process dataset (feature extraction + label encoding) ---
    # the config lives inside the subcache it describes, so each cache is
    # self-validating and they cannot invalidate one another
    processed_config = ProcessedDatasetConfig.from_args(args, vocab_size)
    processed_config_path = os.path.join(processed_path, "config.yaml")
    processed_config.check_cached(processed_config_path)

    dataset = prepare_dataset_for_training(
        dataset,
        processor,
        model_type=args.model.type,
        sampling_rate=args.audio.sampling_rate,
        max_audio_length_seconds=args.dataset.get("max_audio_length_seconds"),
        max_label_length=args.dataset.get("max_label_length"),
        processed_path=processed_path,
    )

    processed_config.save(processed_config_path)

    # --- Stage 4: Setup model ---
    model = setup_model(args, processor)

    # --- Stage 5: Train ---
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

    # copy config to best checkpoint for reproducibility
    with open(os.path.join(best_model_dir, "training_config.yaml"), "w") as f:
        OmegaConf.save(args, f)

    print(f"Training complete. Best model saved to {best_model_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
