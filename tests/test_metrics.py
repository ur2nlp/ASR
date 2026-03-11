"""Tests for src/metrics.py."""

import numpy as np
import torch

from src.metrics import preprocess_logits_for_metrics


class TestPreprocessLogitsForMetrics:
    def test_reduces_to_pred_ids(self):
        # (batch=2, seq_len=3, vocab_size=5)
        logits = torch.randn(2, 3, 5)
        labels = torch.zeros(2, 3)
        result = preprocess_logits_for_metrics(logits, labels)
        assert result.shape == (2, 3)

    def test_argmax_correctness(self):
        logits = torch.tensor([[[0.1, 0.9, 0.0], [0.8, 0.1, 0.1]]])
        labels = torch.zeros(1, 2)
        result = preprocess_logits_for_metrics(logits, labels)
        assert result[0, 0].item() == 1
        assert result[0, 1].item() == 0
