"""Tests for the torch-free parts of tools/align_long_spans.py."""

from tools.align_long_spans import (
    Span,
    WordTiming,
    build_pseudo_spans,
    evaluate_pseudo_span,
    normalize_for_alignment,
    plan_span,
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
        groups = build_pseudo_spans(spans, min_seconds=40.0, max_seconds=90.0, max_gap=1.0)
        assert [len(group) for group in groups] == [4, 4]

    def test_short_no_speech_gaps_are_joined(self):
        # the real layout: a <no-speech> span between every pair of utterances
        spans = [
            Span(0.0, 12.0, ["a"]),
            Span(12.5, 25.0, ["b"]),
            Span(25.3, 38.0, ["c"]),
            Span(38.4, 50.0, ["d"]),
        ]
        groups = build_pseudo_spans(spans, min_seconds=40.0, max_seconds=90.0, max_gap=1.0)
        assert [len(group) for group in groups] == [4]

    def test_gap_breaks_a_group(self):
        spans = [
            Span(0.0, 25.0, ["a"]),
            Span(30.0, 55.0, ["b"]),
            Span(55.0, 80.0, ["c"]),
        ]
        groups = build_pseudo_spans(spans, min_seconds=40.0, max_seconds=90.0, max_gap=1.0)
        assert len(groups) == 1
        assert groups[0][0].start == 30.0

    def test_long_span_is_never_a_member(self):
        spans = [
            Span(0.0, 20.0, ["a"]),
            Span(20.0, 60.0, ["long"]),
            Span(60.0, 80.0, ["b"]),
        ]
        assert build_pseudo_spans(spans, min_seconds=30.0, max_seconds=90.0, max_gap=1.0) == []

    def test_respects_max_length(self):
        spans = [Span(0.0, 20.0, ["a"]), Span(20.0, 45.0, ["b"]), Span(45.0, 70.0, ["c"])]
        groups = build_pseudo_spans(spans, min_seconds=60.0, max_seconds=50.0, max_gap=1.0)
        assert groups == []


class TestEvaluatePseudoSpan:
    def test_scores_words_and_boundaries(self):
        members = [Span(0.0, 5.0, ["a", "b"]), Span(5.0, 10.0, ["c"])]
        timings = [timed("a", 0.5, 1.0), timed("b", 5.5, 6.0), timed("c", 7.0, 8.0)]
        word_rows, boundary_rows = evaluate_pseudo_span(members, timings)
        assert [row["correct"] for row in word_rows] == [True, False, True]
        assert boundary_rows[0]["predicted"] == 6.5
        assert boundary_rows[0]["error"] == 1.5

    def test_prediction_inside_no_speech_gap_is_exact(self):
        members = [Span(0.0, 5.0, ["a"]), Span(7.0, 10.0, ["b"])]
        timings = [timed("a", 1.0, 4.0), timed("b", 8.0, 9.0)]
        _, boundary_rows = evaluate_pseudo_span(members, timings)
        assert boundary_rows[0]["predicted"] == 6.0
        assert boundary_rows[0]["error"] == 0.0

    def test_early_prediction_has_negative_error(self):
        members = [Span(0.0, 5.0, ["a"]), Span(7.0, 10.0, ["b"])]
        timings = [timed("a", 1.0, 2.0), timed("b", 4.0, 9.0)]
        _, boundary_rows = evaluate_pseudo_span(members, timings)
        assert boundary_rows[0]["error"] == -2.0

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
    def split(self, timings, span_start, span_end, max_seconds=30.0, min_pause=0.3):
        return split_at_pauses(
            timings,
            span_start,
            span_end,
            max_seconds=max_seconds,
            min_seconds=2.0,
            min_pause=min_pause,
        )

    def test_short_span_is_one_piece(self):
        timings = [timed("a", 0.5, 1.0), timed("b", 2.0, 3.0)]
        pieces = self.split(timings, 0.0, 4.0)
        assert [(piece.first_word, piece.end_word) for piece in pieces] == [(0, 2)]
        assert (pieces[0].start, pieces[0].end) == (0.0, 4.0)
        assert pieces[0].pause_before is None

    def test_cuts_at_midpoint_of_longest_pause(self):
        timings = [
            timed("a", 0.0, 10.0),
            timed("b", 10.5, 20.0),
            timed("c", 25.0, 35.0),
            timed("d", 35.4, 40.0),
        ]
        pieces = self.split(timings, 0.0, 41.0)
        assert [(piece.first_word, piece.end_word) for piece in pieces] == [(0, 2), (2, 4)]
        assert pieces[0].end == pieces[1].start == 22.5
        assert (pieces[0].start, pieces[1].end) == (0.0, 41.0)
        assert pieces[1].pause_before == 5.0

    def test_pauses_below_threshold_are_never_cut(self):
        timings = [timed(str(index), 2.0 * index, 2.0 * index + 1.9) for index in range(30)]
        pieces = self.split(timings, 0.0, 60.0, min_pause=0.3)
        assert len(pieces) == 1
        assert pieces[0].end - pieces[0].start > 30.0

    def test_every_piece_fits_when_pauses_allow(self):
        timings = [timed(str(index), 3.0 * index, 3.0 * index + 2.5) for index in range(40)]
        pieces = self.split(timings, 0.0, 120.0, max_seconds=20.0)
        assert pieces[0].first_word == 0 and pieces[-1].end_word == 40
        for previous, following in zip(pieces, pieces[1:]):
            assert previous.end_word == following.first_word
            assert previous.end == following.start
        for piece in pieces:
            assert piece.end - piece.start <= 20.0

    def test_piece_duration_includes_span_edges(self):
        # word extents fit in 30 s, but the span's leading silence pushes the audio over
        timings = [timed("a", 10.0, 20.0), timed("b", 21.0, 35.0)]
        pieces = self.split(timings, 0.0, 35.0)
        assert len(pieces) == 2
        assert pieces[0].end == 20.5

    def test_unaligned_word_stays_with_predecessor(self):
        timings = [
            timed("a", 0.0, 10.0),
            WordTiming("1", None, None, None),
            timed("b", 25.0, 35.0),
        ]
        pieces = self.split(timings, 0.0, 35.0)
        assert [(piece.first_word, piece.end_word) for piece in pieces] == [(0, 2), (2, 3)]

    def test_nothing_aligned_is_one_piece(self):
        timings = [WordTiming("1", None, None, None)]
        pieces = self.split(timings, 0.0, 50.0)
        assert [(piece.first_word, piece.end_word) for piece in pieces] == [(0, 1)]


class TestPlanSpan:
    def plan(self, span, timings, min_span_score=0.35):
        return plan_span(
            span,
            timings,
            max_seconds=30.0,
            min_piece_seconds=2.0,
            min_pause=0.3,
            min_span_score=min_span_score,
        )

    def test_ok_pieces_keep_the_transcribers_words(self):
        span = Span(0.0, 41.0, ["Sawubona,", "baba", "Unjani?", "Ngiyaphila."], index=7)
        timings = [
            timed(word, start, end)
            for word, (start, end) in zip(
                span.words, [(0.0, 10.0), (10.5, 20.0), (25.0, 35.0), (35.4, 40.0)]
            )
        ]
        rows = self.plan(span, timings)
        assert [row["status"] for row in rows] == ["ok", "ok"]
        assert [row["text"] for row in rows] == ["Sawubona, baba", "Unjani? Ngiyaphila."]
        assert {row["span_index"] for row in rows} == {7}

    def test_low_span_score_marks_every_piece(self):
        span = Span(0.0, 41.0, ["a", "b"])
        timings = [
            WordTiming("a", 0.0, 10.0, 0.1),
            WordTiming("b", 25.0, 35.0, 0.2),
        ]
        rows = self.plan(span, timings)
        assert {row["status"] for row in rows} == {"low_score"}

    def test_uncuttable_piece_is_too_long(self):
        span = Span(0.0, 60.0, [str(index) for index in range(30)])
        timings = [timed(str(index), 2.0 * index, 2.0 * index + 1.9) for index in range(30)]
        rows = self.plan(span, timings)
        assert [row["status"] for row in rows] == ["too_long"]
