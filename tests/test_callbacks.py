"""Tests for src/callbacks.py."""

import math
from unittest.mock import MagicMock

import pytest

from src.callbacks import (
    DelayedEarlyStoppingCallback,
    DetectBrokenLossCallback,
)


class TestDetectBrokenLossCallback:
    def test_nan_loss_raises(self):
        callback = DetectBrokenLossCallback()
        state = MagicMock()
        state.global_step = 100
        with pytest.raises(RuntimeError, match="Training loss"):
            callback.on_log(
                args=None, state=state, control=None,
                logs={"loss": float("nan")},
            )

    def test_inf_loss_raises(self):
        callback = DetectBrokenLossCallback()
        state = MagicMock()
        state.global_step = 50
        with pytest.raises(RuntimeError, match="Training loss"):
            callback.on_log(
                args=None, state=state, control=None,
                logs={"loss": float("inf")},
            )

    def test_normal_loss_no_error(self):
        callback = DetectBrokenLossCallback()
        state = MagicMock()
        result = callback.on_log(
            args=None, state=state, control=MagicMock(),
            logs={"loss": 2.5},
        )
        assert result is not None

    def test_no_logs_no_error(self):
        callback = DetectBrokenLossCallback()
        result = callback.on_log(
            args=None, state=MagicMock(), control=MagicMock(),
            logs=None,
        )
        assert result is not None


class TestDelayedEarlyStoppingCallback:
    def test_delay_prevents_early_activation(self):
        callback = DelayedEarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_delay_steps=5000,
        )
        state = MagicMock()
        state.global_step = 2000
        control = MagicMock()

        callback.on_evaluate(
            args=MagicMock(), state=state, control=control,
            metrics={"eval_wer": 0.5},
        )
        assert not callback.delay_passed

    def test_activates_after_delay(self):
        callback = DelayedEarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_delay_steps=5000,
        )
        state = MagicMock()
        state.global_step = 6000
        control = MagicMock()

        callback.on_evaluate(
            args=MagicMock(), state=state, control=control,
            metrics={"eval_wer": 0.5},
        )
        assert callback.delay_passed
