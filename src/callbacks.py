"""Custom Trainer callbacks for ASR fine-tuning.

Includes freeze/unfreeze scheduling, delayed early stopping (to avoid
premature stopping in the early unstable training phase), and a broken
loss detector.
"""

import math
import sys

from transformers import (
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from src.models import _freeze_feature_extractor, unfreeze_all


class FeatureExtractorFreezeCallback(TrainerCallback):
    """Freeze the feature extractor at the start of training.

    This is redundant if freeze_feature_extractor=true in the training config,
    but useful when you want to unfreeze later via UnfreezeCallback.
    """

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        _freeze_feature_extractor(model)
        return control


class LayerFreezeCallback(TrainerCallback):
    """Freeze the first N encoder layers at training start."""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        from src.models import _freeze_encoder_layers
        _freeze_encoder_layers(model, self.num_layers)
        return control


class UnfreezeCallback(TrainerCallback):
    """Unfreeze all model parameters at a specified training step.

    Args:
        unfreeze_step: The global step at which to unfreeze all parameters.
    """

    def __init__(self, unfreeze_step: int):
        self.unfreeze_step = unfreeze_step
        self.already_unfrozen = False

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        if self.already_unfrozen:
            return control

        if state.global_step >= self.unfreeze_step:
            unfreeze_all(model)
            self.already_unfrozen = True
            print(
                f"UnfreezeCallback: unfroze all parameters at step {state.global_step}",
                file=sys.stderr,
            )

        return control


class DelayedEarlyStoppingCallback(EarlyStoppingCallback):
    """Early stopping that only activates after a delay period.

    Prevents premature stopping during the initial unstable training phase.

    Args:
        early_stopping_patience: Number of evaluations without improvement
            before stopping.
        early_stopping_delay_steps: Number of steps to wait before activating
            early stopping.
        early_stopping_threshold: Minimum improvement to qualify as an
            improvement.
    """

    def __init__(
        self,
        early_stopping_patience: int = 5,
        early_stopping_delay_steps: int = 0,
        early_stopping_threshold: float = 0.0,
    ):
        super().__init__(
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold,
        )
        self.delay_steps = early_stopping_delay_steps
        self.delay_passed = False

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not self.delay_passed:
            if state.global_step >= self.delay_steps:
                self.delay_passed = True
                print(
                    f"DelayedEarlyStopping: activated at step {state.global_step}",
                    file=sys.stderr,
                )
            else:
                return control

        return super().on_evaluate(args, state, control, metrics=metrics, **kwargs)


class DetectBrokenLossCallback(TrainerCallback):
    """Stop training if loss becomes NaN or diverges."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return control

        loss = logs.get("loss")
        if loss is not None:
            if math.isnan(loss) or math.isinf(loss):
                raise RuntimeError(
                    f"Training loss is {loss} at step {state.global_step}. "
                    f"This usually indicates a learning rate that is too high "
                    f"or numerical instability."
                )

        return control
