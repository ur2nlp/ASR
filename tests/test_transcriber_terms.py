"""Tests for tools/transcriber_terms.py."""

from collections import Counter

from tools.transcriber_terms import (
    dispersion,
    excess_by_word,
    group_by_support,
    label_tokens,
    log_odds_scores,
    supporting_words,
    tfidf_scores,
    tokenize,
    word_ngrams,
)


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


def test_label_tokens_match_training_labels():
    # annotations dropped, lowercased, punctuation removed except the straight apostrophe
    line = "Ngithi [inaudible] (mhm) K'thiwa... yebo, Hhayi!"
    assert label_tokens(line) == ["ngithi", "k'thiwa", "yebo", "hhayi"]


def test_word_ngrams_mark_edges_and_stay_within_the_word():
    assert word_ngrams("ngi", 4) == ["_ngi", "ngi_"]
    assert word_ngrams("a", 4) == []


class TestDispersion:
    # equal-sized transcribers, so excess = own count - other count
    def excess(self, own, other, words):
        return excess_by_word("x", words, Counter(own), Counter(other), 100, 100)

    def test_one_distinctive_word_gives_about_one_effective_word(self):
        excess = self.excess(
            {"mhlampe": 30, "lampa": 1, "ampela": 1},
            {"mhlampe": 2, "lampa": 1, "ampela": 1},
            {"mhlampe": 1, "lampa": 1, "ampela": 1},
        )
        effective, top_word, share = dispersion(excess)
        assert top_word == "mhlampe" and share == 1.0 and effective == 1.0

    def test_spread_excess_gives_many_effective_words(self):
        words = {f"w{index}": 1 for index in range(10)}
        excess = self.excess({w: 3 for w in words}, {w: 1 for w in words}, words)
        effective, _, share = dispersion(excess)
        assert abs(effective - 10.0) < 1e-9
        assert abs(share - 0.1) < 1e-9

    def test_underused_words_do_not_count(self):
        excess = self.excess({"a": 5, "b": 0}, {"a": 1, "b": 9}, {"a": 1, "b": 1})
        assert set(excess) == {"a"}


def test_supporting_words_cover_the_share():
    excess = {"a": 50.0, "b": 30.0, "c": 15.0, "d": 5.0}
    assert supporting_words(excess, 0.8) == {"a", "b"}
    assert supporting_words(excess, 0.9) == {"a", "b", "c"}


def test_grouping_collapses_ngrams_of_one_word():
    ranked = [
        ("hlam", {"mhlampe"}),
        ("_ngb", {"ngbona", "ngbuza", "ngbheka"}),
        ("ampe", {"mhlampe"}),
        ("ngbo", {"ngbona", "ngbonile"}),
    ]
    assert group_by_support(ranked, 0.5) == [0, 1, 0, 2]
    assert group_by_support(ranked, 0.25) == [0, 1, 0, 1]
