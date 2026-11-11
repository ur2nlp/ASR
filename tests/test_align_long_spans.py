"""Tests for the torch-free parts of tools/align_long_spans.py."""

from tools.align_long_spans import (
    Span,
    WordTiming,
    build_pseudo_spans,
    evaluate_pseudo_span,
    normalize_for_alignment,
    split_at_pauses,
)


def timed(word: str, start: float, end: float) -> WordTiming:
    return WordTiming(word=word, start=start, end=end, score=0.9)


class TestNormalizeForAlignment:
    def test_lowercases_and_strips_punctuation(self):
        assert normalize_for_alignment("Ngiyabonga,") == "ngiyabonga"

    def test_strips_diacritics(self):
        assert normalize_for_alignment("café") == "cafe"

    def test_keeps_inner_apostrophe_and_maps_curly_variant(self):
        assert normalize_for_alignment("don’t") == "don't"

    def test_digits_only_word_becomes_empty(self):
        assert normalize_for_alignment("2024") == ""


class TestBuildPseudoSpans:
    def test_groups_contiguous_spans_until_min_length(self):
        spans = [Span(10.0 * index, 10.0 * (index + 1), ["w"]) for index in range(9)]
        groups = build_pseudo_spans(spans, min_seconds=40.0, max_seconds=90.0)
        assert [len(group) for group in groups] == [4, 4]

    def test_gap_breaks_a_group(self):
        spans = [
            Span(0.0, 25.0, ["a"]),
            Span(30.0, 55.0, ["b"]),
            Span(55.0, 80.0, ["c"]),
        ]
        groups = build_pseudo_spans(spans, min_seconds=40.0, max_seconds=90.0)
        assert len(groups) == 1
        assert groups[0][0].start == 30.0

    def test_long_span_is_never_a_member(self):
        spans = [
            Span(0.0, 20.0, ["a"]),
            Span(20.0, 60.0, ["long"]),
            Span(60.0, 80.0, ["b"]),
        ]
        assert build_pseudo_spans(spans, min_seconds=30.0, max_seconds=90.0) == []

    def test_respects_max_length(self):
        spans = [Span(0.0, 20.0, ["a"]), Span(20.0, 45.0, ["b"]), Span(45.0, 70.0, ["c"])]
        groups = build_pseudo_spans(spans, min_seconds=60.0, max_seconds=50.0)
        assert groups == []


class TestEvaluatePseudoSpan:
    def test_scores_words_and_boundaries(self):
        members = [Span(0.0, 5.0, ["a", "b"]), Span(5.0, 10.0, ["c"])]
        timings = [timed("a", 0.5, 1.0), timed("b", 5.5, 6.0), timed("c", 7.0, 8.0)]
        word_rows, boundary_rows = evaluate_pseudo_span(members, timings)
        assert [row["correct"] for row in word_rows] == [True, False, True]
        assert boundary_rows[0]["predicted"] == 6.5
        assert boundary_rows[0]["error"] == 1.5

    def test_word_aligned_at_time_zero_counts_as_aligned(self):
        members = [Span(0.0, 5.0, ["a"]), Span(5.0, 10.0, ["b"])]
        timings = [timed("a", 0.0, 0.0), timed("b", 6.0, 7.0)]
        _, boundary_rows = evaluate_pseudo_span(members, timings)
        assert boundary_rows[0]["predicted"] == 3.0

    def test_unaligned_word_is_unjudged(self):
        members = [Span(0.0, 5.0, ["1"]), Span(5.0, 10.0, ["b"])]
        timings = [WordTiming("1", None, None, None), timed("b", 6.0, 7.0)]
        word_rows, boundary_rows = evaluate_pseudo_span(members, timings)
        assert word_rows[0]["correct"] is None
        assert boundary_rows[0]["predicted"] is None


class TestSplitAtPauses:
    def test_short_sequence_is_one_piece(self):
        timings = [timed("a", 0.0, 1.0), timed("b", 2.0, 3.0)]
        assert split_at_pauses(timings, max_seconds=30.0, min_seconds=2.0) == [(0, 2)]

    def test_cuts_at_longest_pause(self):
        timings = [
            timed("a", 0.0, 10.0),
            timed("b", 10.5, 20.0),
            timed("c", 25.0, 35.0),
            timed("d", 35.2, 40.0),
        ]
        assert split_at_pauses(timings, max_seconds=30.0, min_seconds=2.0) == [(0, 2), (2, 4)]

    def test_every_piece_fits_when_pauses_allow(self):
        timings = [timed(str(index), 3.0 * index, 3.0 * index + 2.5) for index in range(40)]
        pieces = split_at_pauses(timings, max_seconds=20.0, min_seconds=2.0)
        assert pieces[0][0] == 0 and pieces[-1][1] == 40
        for first, last in pieces:
            assert timings[last - 1].end - timings[first].start <= 20.0

    def test_unaligned_word_stays_with_predecessor(self):
        timings = [
            timed("a", 0.0, 10.0),
            WordTiming("1", None, None, None),
            timed("b", 25.0, 35.0),
        ]
        assert split_at_pauses(timings, max_seconds=30.0, min_seconds=2.0) == [(0, 2), (2, 3)]
