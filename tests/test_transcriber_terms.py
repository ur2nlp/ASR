"""Tests for tools/transcriber_terms.py."""

from collections import Counter

from tools.transcriber_terms import log_odds_scores, tfidf_scores, tokenize


class TestTokenize:
    def test_annotations_are_single_tokens(self):
        assert tokenize("Ngithi [uhm] (ahleke) <no-speech> yebo") == [
            "ngithi",
            "[uhm]",
            "(ahleke)",
            "<no-speech>",
            "yebo",
        ]

    def test_punctuation_runs_and_apostrophes(self):
        assert tokenize("Hayi... don't?") == ["hayi", "...", "don't", "?"]

    def test_keep_case(self):
        assert tokenize("Yebo [Uhm]", keep_case=True) == ["Yebo", "[Uhm]"]


def test_tfidf_zero_for_term_every_document_uses():
    counts = {"A": Counter({"x": 5, "a": 1}), "B": Counter({"x": 1, "b": 1})}
    eligible = {"A": {"x", "a"}, "B": {"x", "b"}}
    scores = tfidf_scores(counts, eligible)
    assert scores["A"]["x"] == 0.0
    assert scores["A"]["a"] > 0.0


def test_log_odds_ranks_overused_shared_term_first():
    # both use "x", but A uses it far more often: TF-IDF is blind to that, log-odds is not
    counts = {
        "A": Counter({"x": 200, "y": 800}),
        "B": Counter({"x": 10, "y": 990}),
    }
    eligible = {"A": {"x", "y"}, "B": {"x", "y"}}
    scores = log_odds_scores(counts, eligible, prior_strength=100.0)
    assert scores["A"]["x"] > 0 > scores["A"]["y"]
    assert scores["B"]["x"] < 0
