"""Forced-align Zulu transcript spans to their audio, to cut long spans at word pauses.

Transcribers timestamp at their own pace, and every span over 30 s is lost to Whisper's
window (27% of the corpus hours; see .claude/long_segments.md). The plan is to align each
long span's cleaned words to its audio with torchaudio's MMS_FA model and cut at pauses.

Two modes:

``align`` aligns every span over 30 s and plans where to cut it, writing a manifest that
``prepare_zulu.py`` can consume. Cuts go only in the middle of an aligned pause of at least
``--min-pause`` (0.3 s): in validation, words either side of such a pause were on the right
side of the true boundary 92-98% of the time, against 31-61% for shorter pauses. The first
piece starts at the span's own start timestamp and the last ends at its end timestamp.
A span is cut at its longest qualifying pause (preferring cuts that leave both sides at
least ``--min-piece-seconds``), recursively, until every piece fits in ``--max-seconds``.
A piece that still does not fit, because it has no qualifying pause, is marked
``too_long`` rather than force-cut. A span whose mean alignment score is below
``--min-span-score`` (0.30) is marked ``low_score`` throughout. With chunked emissions
(job 2359341), every badly misaligned validation pseudo-span scored below 0.30 and those
above it kept >= 79% word accuracy (p10 >= 87%), at 40-90, 120-300 and 200-300 s alike. Outputs (in ``--output-dir``):

- ``manifest.csv``: one row per planned piece -- recording, ``span_index`` (the index
  ``prepare_zulu`` uses in ``{stem}_{index:03d}``), piece number, start/end in the
  recording, word counts, piece and span mean score, the aligned pause at the cut before
  it, ``status`` (``ok``, ``too_long``, ``low_score``, ``align_error``) and the text.
- ``summary.json``: long-span hours and recovered (``ok``) hours, overall and per
  transcriber, and hours by status.
- ``sample/`` (with ``--sample-per-transcriber``): random ``ok`` pieces as ``.wav`` +
  ``.txt`` for a listening check.

``validate`` measures how well alignment works on data whose answer is known: runs of consecutive *short*, contiguous spans are joined into
pseudo-long spans (40-90 s by default), aligned as one span, and the predicted boundaries
are compared with the transcriber's real timestamps. Transcribers put a ``<no-speech>`` span
between most utterances, so members may be separated by such gaps (up to ``--max-gap``);
the gap audio stays in the clip, and a predicted boundary anywhere inside it is exact. Outputs (in ``--output-dir``):

- ``words.csv``: every word with its aligned time, alignment score, and whether its
  midpoint fell inside the span it truly belongs to.
- ``boundaries.csv``: every true internal timestamp with the predicted boundary (midpoint
  of the pause between the neighbouring spans' aligned words) and the error in seconds.
- ``pseudo_spans.csv``: per pseudo-span mean alignment score and word accuracy, for
  choosing a score threshold below which cuts should be discarded.
- ``summary.json``: the same, aggregated per transcriber.

Alignment text: each cleaned word (``prepare_zulu.clean_text``) is lowercased, stripped of
diacritics, and reduced to MMS_FA's alphabet (a-z and the apostrophe). A word with nothing
left (digits, pure punctuation) is not aligned and has no time. Only alignment uses the
reduced form; labels always keep the transcriber's words.

GPU work: run on a compute node, never on circ's login node (see .claude/CLAUDE.md).

Usage:
    python -m tools.align_long_spans align --output-dir outputs/long_span_alignment \
        --sample-per-transcriber 5
    python -m tools.align_long_spans validate --raw-dir data/zulu/raw \\
        --metadata data/zulu/metadata.csv --output-dir outputs/alignment_validation
"""

import argparse
import csv
import json
import os
import random
import statistics
import sys
import unicodedata
from dataclasses import dataclass

from tools.prepare_zulu import TARGET_SAMPLING_RATE, clean_text, parse_segments

MMS_ALPHABET = set("abcdefghijklmnopqrstuvwxyz'")
APOSTROPHE_VARIANTS = {"’": "'", "‘": "'", "`": "'"}
STAR = "*"
LONG_SPAN_SECONDS = 30.0
# wav2vec2's convolutional front end: one frame per 320 samples, each seeing 400
FRAME_STRIDE_SAMPLES = 320
FRAME_WINDOW_SAMPLES = 400


@dataclass
class Span:
    """One transcriber span: absolute times in the recording and its cleaned words."""

    start: float
    end: float
    words: list[str]
    # position in parse_segments' output, i.e. prepare_zulu's `{stem}_{index:03d}`
    index: int = -1


@dataclass
class WordTiming:
    """A word's aligned time (absolute seconds) and mean token probability.

    All three are None for a word with no alignable characters.
    """

    word: str
    start: float | None
    end: float | None
    score: float | None


# ---------------------------------------------------------------------------
# Text and span preparation (no torch)
# ---------------------------------------------------------------------------


def normalize_for_alignment(word: str) -> str:
    """Reduce a word to MMS_FA's alphabet: lowercase a-z and the apostrophe.

    Args:
        word: One whitespace-delimited word of cleaned transcript text.

    Returns:
        The reduced form, possibly empty.
    """
    for variant, replacement in APOSTROPHE_VARIANTS.items():
        word = word.replace(variant, replacement)
    decomposed = unicodedata.normalize("NFKD", word.lower())
    kept = [character for character in decomposed if character in MMS_ALPHABET]
    return "".join(kept).strip("'")


def load_spans(transcript_path: str) -> list[Span]:
    """Parse a transcript into spans with cleaned, non-empty word lists."""
    spans = []
    for index, (start, end, raw_text) in enumerate(parse_segments(transcript_path)):
        words = clean_text(raw_text).split()
        if words:
            spans.append(Span(start=start, end=end, words=words, index=index))
    return spans


def build_pseudo_spans(
    spans: list[Span],
    min_seconds: float,
    max_seconds: float,
    max_gap: float,
) -> list[list[Span]]:
    """Group consecutive short spans into pseudo-long spans.

    Consecutive spans may be separated by a gap of at most ``max_gap`` seconds: the dropped
    ``<no-speech>`` (or markup-only) span between them, whose audio stays in the clip. A
    group is closed as soon as it reaches ``min_seconds``; a span that would push it past
    ``max_seconds``, a longer gap, or a span already over the 30 s limit starts a new group.
    Groups shorter than ``min_seconds`` or with a single member are discarded.

    Args:
        spans: A recording's spans in file order.
        min_seconds: Minimum pseudo-span duration.
        max_seconds: Maximum pseudo-span duration.
        max_gap: Longest gap allowed between consecutive members.

    Returns:
        A list of groups, each a list of at least two spans.
    """
    groups: list[list[Span]] = []
    current: list[Span] = []

    def close_current() -> None:
        duration = current[-1].end - current[0].start if current else 0.0
        if len(current) >= 2 and duration >= min_seconds:
            groups.append(list(current))
        current.clear()

    for span in spans:
        if span.end - span.start > LONG_SPAN_SECONDS:
            close_current()
            continue
        if current:
            gap = span.start - current[-1].end
            too_long = span.end - current[0].start > max_seconds
            if gap < -1e-6 or gap > max_gap or too_long:
                close_current()
        current.append(span)
        if current[-1].end - current[0].start >= min_seconds:
            close_current()
    close_current()
    return groups


@dataclass
class Piece:
    """A planned cut of one span: a word range and the audio interval it maps to."""

    first_word: int
    end_word: int
    start: float
    end: float
    # aligned pause at the cut that starts this piece; None for the span's first piece
    pause_before: float | None


def split_at_pauses(
    timings: list[WordTiming],
    span_start: float,
    span_end: float,
    max_seconds: float,
    min_seconds: float,
    min_pause: float,
) -> list[Piece]:
    """Recursively cut a span at its longest qualifying pause until every piece fits.

    Only pauses of at least ``min_pause`` between two consecutive aligned words qualify,
    and the cut is placed at the pause's midpoint. Pieces' audio runs from cut to cut, with
    the span's own ``span_start``/``span_end`` at the outer edges, and it is that audio
    duration which must fit in ``max_seconds``. Among qualifying pauses, ones leaving both
    sides at least ``min_seconds`` are preferred, then the longest. Unaligned words stay
    with the preceding aligned word (or the first piece, if none precedes them).

    Args:
        timings: Aligned words of one span, in order, with absolute times.
        span_start: The span's start timestamp.
        span_end: The span's end timestamp.
        max_seconds: Longest allowed piece.
        min_seconds: Shortest preferred piece.
        min_pause: Shortest pause a cut may be placed in.

    Returns:
        Pieces covering every word, in order. A piece may still exceed ``max_seconds`` if
        it has no qualifying pause; the caller decides what to do with it.
    """
    aligned = [index for index, timing in enumerate(timings) if timing.start is not None]
    if not aligned:
        return [Piece(0, len(timings), span_start, span_end, None)]

    def pause_at(position: int) -> float:
        """Pause before aligned word `position` (between it and the previous one)."""
        return timings[aligned[position]].start - timings[aligned[position - 1]].end

    def cut_time(position: int) -> float:
        return (timings[aligned[position - 1]].end + timings[aligned[position]].start) / 2

    def cut(first: int, last: int, left_time: float, right_time: float) -> list[int]:
        """Return the positions in `aligned` where new pieces start, within first..last."""
        if right_time - left_time <= max_seconds:
            return []
        candidates = []
        for position in range(first + 1, last + 1):
            pause = pause_at(position)
            if pause < min_pause:
                continue
            time = cut_time(position)
            balanced = time - left_time >= min_seconds and right_time - time >= min_seconds
            candidates.append((balanced, pause, position))
        if not candidates:
            return []
        _, _, best = max(candidates)
        best_time = cut_time(best)
        left_cuts = cut(first, best - 1, left_time, best_time)
        right_cuts = cut(best, last, best_time, right_time)
        return left_cuts + [best] + right_cuts

    positions = cut(0, len(aligned) - 1, span_start, span_end)
    starts = [0] + [aligned[position] for position in positions]
    ends = starts[1:] + [len(timings)]
    times = [span_start] + [cut_time(position) for position in positions] + [span_end]
    pauses = [None] + [pause_at(position) for position in positions]

    pieces = []
    for piece_index in range(len(starts)):
        pieces.append(
            Piece(
                first_word=starts[piece_index],
                end_word=ends[piece_index],
                start=times[piece_index],
                end=times[piece_index + 1],
                pause_before=pauses[piece_index],
            )
        )
    return pieces


# ---------------------------------------------------------------------------
# Evaluation against known boundaries (no torch)
# ---------------------------------------------------------------------------


def evaluate_pseudo_span(
    members: list[Span],
    timings: list[WordTiming],
) -> tuple[list[dict], list[dict]]:
    """Compare aligned word times with the members' true span boundaries.

    Args:
        members: The spans joined into this pseudo-span.
        timings: Aligned times for the concatenation of every member's words.

    Returns:
        ``(word_rows, boundary_rows)``. A word is correct when its aligned midpoint lies in
        its own member's ``[start, end]``. A true boundary is the interval between one
        member's end and the next one's start (often a ``<no-speech>`` gap, possibly
        empty). The prediction is the midpoint between the last aligned word before it
        and the first aligned word after it, and its error is the signed distance to that
        interval: 0 inside it, negative before it, positive after it. Prediction and error
        are None when either side has no aligned word.
    """
    word_rows = []
    member_timings = []
    position = 0
    for member_index, member in enumerate(members):
        own = timings[position : position + len(member.words)]
        position += len(member.words)
        member_timings.append(own)
        for timing in own:
            correct = None
            if timing.start is not None:
                midpoint = (timing.start + timing.end) / 2
                correct = member.start <= midpoint <= member.end
            word_rows.append(
                {
                    "member": member_index,
                    "word": timing.word,
                    "start": timing.start,
                    "end": timing.end,
                    "score": timing.score,
                    "true_start": member.start,
                    "true_end": member.end,
                    "correct": correct,
                }
            )

    boundary_rows = []
    for member_index in range(1, len(members)):
        before = [timing for timing in member_timings[member_index - 1] if timing.end is not None]
        after = [timing for timing in member_timings[member_index] if timing.start is not None]
        true_start = members[member_index - 1].end
        true_end = members[member_index].start
        predicted = None
        error = None
        aligned_gap = None
        if before and after:
            aligned_gap = after[0].start - before[-1].end
            predicted = (before[-1].end + after[0].start) / 2
            if predicted < true_start:
                error = predicted - true_start
            elif predicted > true_end:
                error = predicted - true_end
            else:
                error = 0.0
        boundary_rows.append(
            {
                "member": member_index,
                "true_start": true_start,
                "true_end": true_end,
                "predicted": predicted,
                "error": error,
                "aligned_gap": aligned_gap,
            }
        )
    return word_rows, boundary_rows


# ---------------------------------------------------------------------------
# Alignment (torch; compute node only)
# ---------------------------------------------------------------------------


@dataclass
class EmissionChunk:
    """One window of audio to run the acoustic model on, and which frames to keep.

    The model sees ``[padded_start, padded_end)`` samples; of its output frames, the
    ``keep_count`` frames from ``keep_from`` on cover the chunk's own core region (None
    keeps every frame to the end). Every boundary is a multiple of the frame stride, so
    frame ``k`` of a window is frame ``padded_start / stride + k`` of the whole clip.
    """

    padded_start: int
    padded_end: int
    keep_from: int
    keep_count: int | None


def plan_emission_chunks(
    num_samples: int,
    chunk_samples: int,
    context_samples: int,
) -> list[EmissionChunk]:
    """Split a clip into roughly equal cores of about `chunk_samples`, each with context.

    The number of cores is ``round(num_samples / chunk_samples)`` (at least 1), so cores
    run from 2/3 to 3/2 of the target and no sliver is left at the end. Concatenating the
    kept frames gives exactly as many frames as running the model on the whole clip.

    Args:
        num_samples: Clip length in samples.
        chunk_samples: Target core length in samples; 0 or less disables chunking.
        context_samples: Extra audio the model sees on each side of a core.

    Returns:
        The chunks in order.
    """
    stride = FRAME_STRIDE_SAMPLES
    if chunk_samples <= 0:
        return [EmissionChunk(0, num_samples, 0, None)]
    num_chunks = max(1, round(num_samples / chunk_samples))
    context = (context_samples // stride) * stride
    # core boundaries on the frame grid; the last core runs to the end of the clip
    boundaries = [
        round(index * num_samples / num_chunks / stride) * stride for index in range(num_chunks)
    ]
    boundaries.append(num_samples)

    chunks = []
    for index in range(num_chunks):
        core_start = boundaries[index]
        core_end = boundaries[index + 1]
        padded_start = max(0, core_start - context)
        is_last = index == num_chunks - 1
        padded_end = num_samples if is_last else min(num_samples, core_end + context)
        keep_count = None if is_last else (core_end - core_start) // stride
        chunks.append(
            EmissionChunk(
                padded_start=padded_start,
                padded_end=padded_end,
                keep_from=(core_start - padded_start) // stride,
                keep_count=keep_count,
            )
        )
    return chunks


def num_frames(num_samples: int) -> int:
    """Frames wav2vec2's front end produces for a clip of `num_samples`."""
    if num_samples < FRAME_WINDOW_SAMPLES:
        return 0
    return (num_samples - FRAME_WINDOW_SAMPLES) // FRAME_STRIDE_SAMPLES + 1


class MmsAligner:
    """Thin wrapper over torchaudio's MMS_FA bundle (model, tokenizer, aligner).

    Verified on circ 2026-11-11: torch 2.14.0, torchaudio 2.11.0 still ship
    ``torchaudio.pipelines.MMS_FA`` and ``torchaudio.functional.forced_align``.
    """

    def __init__(
        self,
        device: str,
        star_edges: bool,
        chunk_seconds: float,
        context_seconds: float,
    ) -> None:
        import torch
        from torchaudio.pipelines import MMS_FA

        self.torch = torch
        self.device = device
        self.star_edges = star_edges
        self.chunk_samples = int(chunk_seconds * TARGET_SAMPLING_RATE)
        self.context_samples = int(context_seconds * TARGET_SAMPLING_RATE)
        self.model = MMS_FA.get_model(with_star=True).to(device).eval()
        self.tokenizer = MMS_FA.get_tokenizer()
        self.aligner = MMS_FA.get_aligner()
        assert MMS_FA.sample_rate == TARGET_SAMPLING_RATE

    def align(self, waveform, words: list[str], offset: float) -> list[WordTiming]:
        """Align words to a 16 kHz mono clip.

        Args:
            waveform: Clip samples, a 1-D float numpy array at 16 kHz.
            words: The transcriber's words, in order.
            offset: The clip's start time in the recording, added to every time.

        Returns:
            One WordTiming per input word.
        """
        normalized = [normalize_for_alignment(word) for word in words]
        alignable = [index for index, form in enumerate(normalized) if form]
        timings = [WordTiming(word, None, None, None) for word in words]
        if not alignable:
            return timings

        transcript = [normalized[index] for index in alignable]
        if self.star_edges:
            transcript = [STAR] + transcript + [STAR]

        with self.torch.inference_mode():
            # (1, num_samples)
            audio = self.torch.from_numpy(waveform).float().unsqueeze(0).to(self.device)
            # (num_frames, num_labels)
            emission = self.emissions(audio)
            token_spans = self.aligner(emission, self.tokenizer(transcript))
        seconds_per_frame = audio.size(1) / emission.size(0) / TARGET_SAMPLING_RATE

        if self.star_edges:
            token_spans = token_spans[1:-1]
        for index, spans in zip(alignable, token_spans):
            num_frames = sum(span.end - span.start for span in spans)
            score = sum(span.score * (span.end - span.start) for span in spans) / num_frames
            timings[index] = WordTiming(
                word=words[index],
                start=offset + spans[0].start * seconds_per_frame,
                end=offset + spans[-1].end * seconds_per_frame,
                score=score,
            )
        return timings

    def emissions(self, audio):
        """Run the model chunk by chunk and stitch the kept frames together.

        MMS_FA's frame probabilities degrade on very long inputs (span score medians
        fell from 0.54 at 30-60 s to 0.21 at > 200 s with word accuracy roughly flat),
        so long clips are scored in ~``chunk_seconds`` windows, each with
        ``context_seconds`` of audio either side that is seen but not kept.

        Args:
            audio: (1, num_samples) tensor on the model's device.

        Returns:
            (num_frames, num_labels) emissions for the whole clip.
        """
        pieces = []
        for chunk in plan_emission_chunks(
            audio.size(1),
            self.chunk_samples,
            self.context_samples,
        ):
            # (1, chunk_frames, num_labels)
            chunk_emission, _ = self.model(audio[:, chunk.padded_start : chunk.padded_end])
            if chunk.keep_count is None:
                kept = chunk_emission[0, chunk.keep_from :]
            else:
                kept = chunk_emission[0, chunk.keep_from : chunk.keep_from + chunk.keep_count]
            pieces.append(kept)
        # (num_frames, num_labels)
        emission = self.torch.cat(pieces, dim=0)
        expected = num_frames(audio.size(1))
        if emission.size(0) != expected:
            raise RuntimeError(
                f"chunked emissions have {emission.size(0)} frames, expected {expected}"
            )
        return emission


# ---------------------------------------------------------------------------
# Validate mode
# ---------------------------------------------------------------------------


def choose_recordings(
    raw_dir: str,
    metadata_path: str,
    per_transcriber: int | None,
    seed: int,
) -> list[tuple[str, str]]:
    """Sample recordings with transcripts, up to `per_transcriber` per transcriber.

    With `per_transcriber` None, every recording is returned.

    Reads only ``{recording}.txt`` names from `raw_dir` and the anonymized metadata.

    Returns:
        ``(transcriber, recording)`` pairs, sorted.
    """
    available = {
        name[:-4]
        for name in os.listdir(raw_dir)
        if name.endswith(".txt") and os.path.isfile(os.path.join(raw_dir, name[:-4] + ".wav"))
    }
    by_transcriber: dict[str, list[str]] = {}
    with open(metadata_path, encoding="utf-8") as metadata_file:
        for row in csv.DictReader(metadata_file):
            if row["recording"] in available and row["language"] != "english":
                by_transcriber.setdefault(row["transcriber"], []).append(row["recording"])

    generator = random.Random(seed)
    chosen = []
    for transcriber in sorted(by_transcriber):
        recordings = sorted(by_transcriber[transcriber])
        generator.shuffle(recordings)
        if per_transcriber is not None:
            recordings = recordings[:per_transcriber]
        for recording in recordings:
            chosen.append((transcriber, recording))
    return sorted(chosen)


def summarize(word_rows: list[dict], boundary_rows: list[dict]) -> dict:
    """Aggregate word accuracy and boundary error for one group of rows."""
    judged = [row["correct"] for row in word_rows if row["correct"] is not None]
    errors = sorted(abs(row["error"]) for row in boundary_rows if row["error"] is not None)

    def quantile(values: list[float], fraction: float) -> float | None:
        if not values:
            return None
        return values[min(len(values) - 1, int(fraction * len(values)))]

    return {
        "words": len(word_rows),
        "words_unaligned": len(word_rows) - len(judged),
        "word_accuracy": sum(judged) / len(judged) if judged else None,
        "boundaries": len(boundary_rows),
        "abs_error_median": quantile(errors, 0.5),
        "abs_error_p90": quantile(errors, 0.9),
        "within_0_5s": sum(error <= 0.5 for error in errors) / len(errors) if errors else None,
    }


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_validate(args: argparse.Namespace) -> None:
    import librosa
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("warning: no GPU; alignment will be slow", file=sys.stderr)
    aligner = MmsAligner(
        device=device,
        star_edges=not args.no_star,
        chunk_seconds=args.chunk_seconds,
        context_seconds=args.context_seconds,
    )

    recordings = choose_recordings(
        args.raw_dir,
        args.metadata,
        args.recordings_per_transcriber,
        args.seed,
    )
    print(f"{len(recordings)} recordings sampled", file=sys.stderr)

    all_words: list[dict] = []
    all_boundaries: list[dict] = []
    pseudo_rows: list[dict] = []
    for transcriber, recording in recordings:
        spans = load_spans(os.path.join(args.raw_dir, f"{recording}.txt"))
        groups = build_pseudo_spans(
            spans,
            args.min_seconds,
            args.max_seconds,
            args.max_gap,
        )
        if not groups:
            continue
        # waveform: (num_samples,) at 16 kHz mono
        waveform, _ = librosa.load(
            os.path.join(args.raw_dir, f"{recording}.wav"),
            sr=TARGET_SAMPLING_RATE,
            mono=True,
        )
        for group_index, members in enumerate(groups):
            start = members[0].start
            end = members[-1].end
            clip = waveform[int(start * TARGET_SAMPLING_RATE) : int(end * TARGET_SAMPLING_RATE)]
            words = [word for member in members for word in member.words]
            try:
                timings = aligner.align(clip, words, offset=start)
            except RuntimeError as error:
                # e.g. more tokens than frames: CTC cannot fit the transcript in the clip
                print(f"warning: {recording} [{start:.1f}, {end:.1f}]: {error}", file=sys.stderr)
                continue
            word_rows, boundary_rows = evaluate_pseudo_span(members, timings)

            identity = {
                "transcriber": transcriber,
                "recording": recording,
                "pseudo_span": group_index,
            }
            word_rows = [{**identity, **row} for row in word_rows]
            boundary_rows = [{**identity, **row} for row in boundary_rows]
            all_words.extend(word_rows)
            all_boundaries.extend(boundary_rows)

            scores = [row["score"] for row in word_rows if row["score"] is not None]
            pseudo_rows.append(
                {
                    **identity,
                    "start": start,
                    "end": end,
                    "members": len(members),
                    "mean_score": statistics.fmean(scores) if scores else None,
                    **summarize(word_rows, boundary_rows),
                }
            )
        print(f"{transcriber} {recording}: {len(groups)} pseudo-span(s)", file=sys.stderr)

    os.makedirs(args.output_dir, exist_ok=True)
    write_csv(os.path.join(args.output_dir, "words.csv"), all_words)
    write_csv(os.path.join(args.output_dir, "boundaries.csv"), all_boundaries)
    write_csv(os.path.join(args.output_dir, "pseudo_spans.csv"), pseudo_rows)

    summary = {"all": summarize(all_words, all_boundaries)}
    for transcriber in sorted({row["transcriber"] for row in all_words}):
        summary[transcriber] = summarize(
            [row for row in all_words if row["transcriber"] == transcriber],
            [row for row in all_boundaries if row["transcriber"] == transcriber],
        )
    summary["settings"] = {key: value for key, value in vars(args).items() if key != "func"}
    with open(os.path.join(args.output_dir, "summary.json"), "w") as summary_file:
        json.dump(summary, summary_file, indent=2)
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# Align mode
# ---------------------------------------------------------------------------


def mean_score(timings: list[WordTiming]) -> float | None:
    scores = [timing.score for timing in timings if timing.score is not None]
    return statistics.fmean(scores) if scores else None


def plan_span(
    span: Span,
    timings: list[WordTiming],
    max_seconds: float,
    min_piece_seconds: float,
    min_pause: float,
    min_span_score: float,
) -> list[dict]:
    """Cut one aligned long span into pieces and give each piece a status.

    Returns:
        One row per piece (see the module docstring for the columns).
    """
    span_score = mean_score(timings)
    pieces = split_at_pauses(
        timings,
        span.start,
        span.end,
        max_seconds,
        min_piece_seconds,
        min_pause,
    )
    rows = []
    for piece_index, piece in enumerate(pieces):
        piece_timings = timings[piece.first_word : piece.end_word]
        if span_score is None or span_score < min_span_score:
            status = "low_score"
        elif piece.end - piece.start > max_seconds:
            status = "too_long"
        else:
            status = "ok"
        piece_score = mean_score(piece_timings)
        rows.append(
            {
                "span_index": span.index,
                "piece": piece_index,
                "start": round(piece.start, 3),
                "end": round(piece.end, 3),
                "duration": round(piece.end - piece.start, 3),
                "words": len(piece_timings),
                "words_aligned": sum(timing.start is not None for timing in piece_timings),
                "piece_score": None if piece_score is None else round(piece_score, 4),
                "span_score": None if span_score is None else round(span_score, 4),
                "pause_before": None
                if piece.pause_before is None
                else round(piece.pause_before, 3),
                "status": status,
                "text": " ".join(span.words[piece.first_word : piece.end_word]),
            }
        )
    return rows


def summarize_manifest(rows: list[dict]) -> dict:
    """Hours by status, overall and per transcriber."""

    def hours(selected: list[dict]) -> float:
        return round(sum(row["duration"] for row in selected) / 3600.0, 3)

    def block(selected: list[dict]) -> dict:
        statuses = sorted({row["status"] for row in selected})
        spans = {(row["recording"], row["span_index"]) for row in selected}
        return {
            "long_spans": len(spans),
            "long_hours": hours(selected),
            "ok_hours": hours([row for row in selected if row["status"] == "ok"]),
            "ok_pieces": sum(row["status"] == "ok" for row in selected),
            "hours_by_status": {
                status: hours([row for row in selected if row["status"] == status])
                for status in statuses
            },
        }

    summary = {"all": block(rows)}
    for transcriber in sorted({row["transcriber"] for row in rows}):
        summary[transcriber] = block([row for row in rows if row["transcriber"] == transcriber])
    return summary


def write_sample(
    rows: list[dict],
    raw_dir: str,
    sample_dir: str,
    per_transcriber: int,
    seed: int,
) -> None:
    """Write random `ok` pieces as .wav/.txt pairs for a listening check."""
    import librosa
    import soundfile

    generator = random.Random(seed)
    os.makedirs(sample_dir, exist_ok=True)
    for transcriber in sorted({row["transcriber"] for row in rows}):
        candidates = [
            row for row in rows if row["transcriber"] == transcriber and row["status"] == "ok"
        ]
        for row in generator.sample(candidates, min(per_transcriber, len(candidates))):
            # clip: (num_samples,) at 16 kHz mono
            clip, _ = librosa.load(
                os.path.join(raw_dir, f"{row['recording']}.wav"),
                sr=TARGET_SAMPLING_RATE,
                mono=True,
                offset=row["start"],
                duration=row["duration"],
            )
            name = f"{transcriber}_{row['recording']}_{row['span_index']:03d}_{row['piece']:02d}"
            soundfile.write(os.path.join(sample_dir, f"{name}.wav"), clip, TARGET_SAMPLING_RATE)
            with open(os.path.join(sample_dir, f"{name}.txt"), "w", encoding="utf-8") as text_file:
                text_file.write(row["text"] + "\n")


def run_align(args: argparse.Namespace) -> None:
    import librosa
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("warning: no GPU; alignment will be slow", file=sys.stderr)
    aligner = MmsAligner(
        device=device,
        star_edges=not args.no_star,
        chunk_seconds=args.chunk_seconds,
        context_seconds=args.context_seconds,
    )

    recordings = choose_recordings(args.raw_dir, args.metadata, None, args.seed)
    rows: list[dict] = []
    for transcriber, recording in recordings:
        long_spans = [
            span
            for span in load_spans(os.path.join(args.raw_dir, f"{recording}.txt"))
            if span.end - span.start > LONG_SPAN_SECONDS
        ]
        if not long_spans:
            continue
        # waveform: (num_samples,) at 16 kHz mono
        waveform, _ = librosa.load(
            os.path.join(args.raw_dir, f"{recording}.wav"),
            sr=TARGET_SAMPLING_RATE,
            mono=True,
        )
        for span in long_spans:
            clip = waveform[
                int(span.start * TARGET_SAMPLING_RATE) : int(span.end * TARGET_SAMPLING_RATE)
            ]
            identity = {"transcriber": transcriber, "recording": recording}
            try:
                timings = aligner.align(clip, span.words, offset=span.start)
            except RuntimeError as error:
                # e.g. CUDA OOM, or more tokens than frames
                print(f"warning: {recording} span {span.index}: {error}", file=sys.stderr)
                if device == "cuda":
                    torch.cuda.empty_cache()
                rows.append(
                    {
                        **identity,
                        "span_index": span.index,
                        "piece": 0,
                        "start": span.start,
                        "end": span.end,
                        "duration": round(span.end - span.start, 3),
                        "words": len(span.words),
                        "words_aligned": 0,
                        "piece_score": None,
                        "span_score": None,
                        "pause_before": None,
                        "status": "align_error",
                        "text": " ".join(span.words),
                    }
                )
                continue
            planned = plan_span(
                span,
                timings,
                args.max_seconds,
                args.min_piece_seconds,
                args.min_pause,
                args.min_span_score,
            )
            rows.extend({**identity, **row} for row in planned)
        print(f"{transcriber} {recording}: {len(long_spans)} long span(s)", file=sys.stderr)

    os.makedirs(args.output_dir, exist_ok=True)
    write_csv(os.path.join(args.output_dir, "manifest.csv"), rows)
    summary = summarize_manifest(rows)
    summary["settings"] = {key: value for key, value in vars(args).items() if key != "func"}
    with open(os.path.join(args.output_dir, "summary.json"), "w") as summary_file:
        json.dump(summary, summary_file, indent=2)
    print(json.dumps(summary, indent=2))

    if args.sample_per_transcriber > 0:
        write_sample(
            rows,
            args.raw_dir,
            os.path.join(args.output_dir, "sample"),
            args.sample_per_transcriber,
            args.seed,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    subparsers = parser.add_subparsers(dest="mode", required=True)

    validate = subparsers.add_parser("validate", help="measure alignment on known boundaries")
    validate.add_argument("--raw-dir", default="data/zulu/raw")
    validate.add_argument("--metadata", default="data/zulu/metadata.csv")
    validate.add_argument("--output-dir", default="outputs/alignment_validation")
    validate.add_argument("--recordings-per-transcriber", type=int, default=10)
    validate.add_argument("--min-seconds", type=float, default=40.0)
    validate.add_argument("--max-seconds", type=float, default=90.0)
    validate.add_argument(
        "--max-gap",
        type=float,
        default=5.0,
        help="longest <no-speech> gap allowed between joined spans, in seconds",
    )
    validate.add_argument(
        "--no-star",
        action="store_true",
        help="do not wrap the transcript in MMS_FA's '*' wildcard token",
    )
    validate.add_argument("--seed", type=int, default=1)
    validate.add_argument(
        "--chunk-seconds",
        type=float,
        default=30.0,
        help="score audio in windows of about this length; 0 scores whole clips at once",
    )
    validate.add_argument(
        "--context-seconds",
        type=float,
        default=2.0,
        help="audio the model sees either side of each window but whose frames are dropped",
    )
    validate.set_defaults(func=run_validate)

    align = subparsers.add_parser("align", help="align long spans and plan their cuts")
    align.add_argument("--raw-dir", default="data/zulu/raw")
    align.add_argument("--metadata", default="data/zulu/metadata.csv")
    align.add_argument("--output-dir", default="outputs/long_span_alignment")
    align.add_argument("--max-seconds", type=float, default=30.0, help="longest piece")
    align.add_argument(
        "--min-piece-seconds",
        type=float,
        default=5.0,
        help="prefer cuts leaving both sides at least this long",
    )
    align.add_argument(
        "--min-pause",
        type=float,
        default=0.3,
        help="shortest aligned pause a cut may be placed in, in seconds",
    )
    align.add_argument(
        "--min-span-score",
        type=float,
        default=0.30,
        help="spans whose mean alignment score is below this are marked low_score",
    )
    align.add_argument(
        "--sample-per-transcriber",
        type=int,
        default=0,
        help="write this many random ok pieces per transcriber to sample/ for listening",
    )
    align.add_argument("--no-star", action="store_true")
    align.add_argument("--seed", type=int, default=1)
    align.add_argument(
        "--chunk-seconds",
        type=float,
        default=30.0,
        help="score audio in windows of about this length; 0 scores whole clips at once",
    )
    align.add_argument(
        "--context-seconds",
        type=float,
        default=2.0,
        help="audio the model sees either side of each window but whose frames are dropped",
    )
    align.set_defaults(func=run_align)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
