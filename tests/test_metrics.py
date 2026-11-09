"""Tests for src/metrics.py."""

import numpy as np
import pytest
import torch

from src.metrics import (
    decode_predictions,
    preprocess_logits_for_metrics,
    utterance_error_counts,
)


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


class TestUtteranceErrorCounts:
    REFERENCES = ["sawubona baba", "ngiyabonga kakhulu impela", "yebo", ""]
    HYPOTHESES = ["sawubona mama", "ngiyabonga impela", "yebo yebo", "noma"]

    def test_counts_one_utterance(self):
        counts = utterance_error_counts("a b c d", "a x c")
        assert counts["ref_words"] == 4
        assert (counts["substitutions"], counts["deletions"], counts["insertions"]) == (1, 1, 0)

    def test_sums_reproduce_corpus_wer_and_cer(self):
        """Summed counts must match the metric the Trainer reports."""
        import evaluate

        counts = [
            utterance_error_counts(reference, hypothesis)
            for reference, hypothesis in zip(self.REFERENCES, self.HYPOTHESES)
        ]
        word_edits = sum(c["substitutions"] + c["deletions"] + c["insertions"] for c in counts)
        wer = word_edits / sum(c["ref_words"] for c in counts)
        cer = sum(c["char_edits"] for c in counts) / sum(c["ref_chars"] for c in counts)

        # compute_metrics drops empty references before scoring
        kept = [(h, r) for h, r in zip(self.HYPOTHESES, self.REFERENCES) if r.strip()]
        predictions, references = [list(side) for side in zip(*kept)]
        expected_wer = evaluate.load("wer").compute(predictions=predictions, references=references)
        expected_cer = evaluate.load("cer").compute(predictions=predictions, references=references)
        assert wer == pytest.approx(expected_wer)
        assert cer == pytest.approx(expected_cer)

    def test_empty_reference_counts_nothing(self):
        counts = utterance_error_counts("", "noma")
        assert counts["ref_words"] == 0
        assert counts["insertions"] == 0


class TestDecodePredictions:
    def test_decodes_without_mutating_inputs(self, vocab_path):
        """The Trainer hands the same arrays back to `predict`'s caller."""
        from src.vocab import create_ctc_tokenizer

        tokenizer = create_ctc_tokenizer(vocab_path)

        class Processor:
            pass

        processor = Processor()
        processor.tokenizer = tokenizer
        labels = np.array([tokenizer("hello").input_ids + [-100]])
        predictions = np.array([tokenizer("hello").input_ids + [-100]])
        labels_before = labels.copy()

        hypotheses, references = decode_predictions(predictions, labels, processor)
        assert references == ["hello"]
        assert hypotheses == ["helo"]
        assert (labels == labels_before).all()
