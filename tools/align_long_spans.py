"""Forced-align Zulu transcript spans to their audio, to cut long spans at word pauses.

Transcribers timestamp at their own pace, and every span over 30 s is lost to Whisper's
window (27% of the corpus hours; see .claude/long_segments.md). The plan is to align each
long span's cleaned words to its audio with torchaudio's MMS_FA model and cut at pauses.

This module currently provides the ``validate`` mode, which measures how well that works on
data whose answer is known: runs of consecutive *short*, contiguous spans are joined into
pseudo-long spans (40-90 s by default), aligned as one span, and the predicted boundaries
are compared with the transcriber's real timestamps. Outputs (in ``--output-dir``):

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


@dataclass
class Span:
    """One transcriber span: absolute times in the recording and its cleaned words."""

    start: float
    end: float
    words: list[str]


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
    for start, end, raw_text in parse_segments(transcript_path):
        words = clean_text(raw_text).split()
        if words:
            spans.append(Span(start=start, end=end, words=words))
    return spans


def build_pseudo_spans(
    spans: list[Span],
    min_seconds: float,
    max_seconds: float,
) -> list[list[Span]]:
    """Group consecutive, contiguous short spans into pseudo-long spans.

    Spans are contiguous when one ends exactly where the next starts, i.e. no dropped
    ``<no-speech>`` or markup-only span lies between them. A group is closed as soon as it
    reaches ``min_seconds``; a span that would push it past ``max_seconds``, a gap, or a
    span already over the 30 s limit starts a new group. Groups shorter than
    ``min_seconds`` or with a single member are discarded.

    Args:
        spans: A recording's spans in file order.
        min_seconds: Minimum pseudo-span duration.
        max_seconds: Maximum pseudo-span duration.

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
            contiguous = abs(span.start - current[-1].end) < 1e-6
            too_long = span.end - current[0].start > max_seconds
            if not contiguous or too_long:
                close_current()
        current.append(span)
        if current[-1].end - current[0].start >= min_seconds:
            close_current()
    close_current()
    return groups


def split_at_pauses(
    timings: list[WordTiming],
    max_seconds: float,
    min_seconds: float,
) -> list[tuple[int, int]]:
    """Recursively cut a word sequence at its longest pause until pieces fit.

    A piece's duration runs from its first to its last aligned word. Cuts are only made
    between two aligned words, and preferably where both sides last at least
    ``min_seconds``; unaligned words stay with the preceding aligned word.

    Args:
        timings: Aligned words of one span, in order.
        max_seconds: Longest allowed piece.
        min_seconds: Shortest preferred piece.

    Returns:
        ``(first, last_exclusive)`` word-index ranges covering every word, in order. A piece
        may still exceed ``max_seconds`` if it has no pause to cut at.
    """
    aligned = [index for index, timing in enumerate(timings) if timing.start is not None]

    def piece_duration(first_aligned: int, last_aligned: int) -> float:
        return timings[aligned[last_aligned]].end - timings[aligned[first_aligned]].start

    def cut(first_aligned: int, last_aligned: int) -> list[int]:
        """Return positions in `aligned` where a new piece starts."""
        if last_aligned <= first_aligned:
            return []
        if piece_duration(first_aligned, last_aligned) <= max_seconds:
            return []
        candidates = []
        for position in range(first_aligned + 1, last_aligned + 1):
            gap = timings[aligned[position]].start - timings[aligned[position - 1]].end
            left = piece_duration(first_aligned, position - 1)
            right = piece_duration(position, last_aligned)
            balanced = left >= min_seconds and right >= min_seconds
            candidates.append((balanced, gap, position))
        _, _, best = max(candidates)
        return cut(first_aligned, best - 1) + [best] + cut(best, last_aligned)

    if not aligned:
        return [(0, len(timings))]
    piece_starts = [0] + [aligned[position] for position in cut(0, len(aligned) - 1)]
    piece_ends = piece_starts[1:] + [len(timings)]
    return list(zip(piece_starts, piece_ends))


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
        its own member's ``[start, end]``. A boundary's prediction is the midpoint between
        the last aligned word before it and the first aligned word after it; it is None
        when either side has no aligned word.
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
        true_boundary = members[member_index].start
        predicted = None
        gap = None
        if before and after:
            gap = after[0].start - before[-1].end
            predicted = (before[-1].end + after[0].start) / 2
        boundary_rows.append(
            {
                "member": member_index,
                "true_boundary": true_boundary,
                "predicted": predicted,
                "error": None if predicted is None else predicted - true_boundary,
                "gap": gap,
            }
        )
    return word_rows, boundary_rows


# ---------------------------------------------------------------------------
# Alignment (torch; compute node only)
# ---------------------------------------------------------------------------


class MmsAligner:
    """Thin wrapper over torchaudio's MMS_FA bundle (model, tokenizer, aligner).

    Verified on circ 2026-11-11: torch 2.14.0, torchaudio 2.11.0 still ship
    ``torchaudio.pipelines.MMS_FA`` and ``torchaudio.functional.forced_align``.
    """

    def __init__(self, device: str, star_edges: bool) -> None:
        import torch
        from torchaudio.pipelines import MMS_FA

        self.torch = torch
        self.device = device
        self.star_edges = star_edges
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
            # (1, num_frames, num_labels)
            emission, _ = self.model(audio)
            token_spans = self.aligner(emission[0], self.tokenizer(transcript))
        seconds_per_frame = audio.size(1) / emission.size(1) / TARGET_SAMPLING_RATE

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


# ---------------------------------------------------------------------------
# Validate mode
# ---------------------------------------------------------------------------


def choose_recordings(
    raw_dir: str,
    metadata_path: str,
    per_transcriber: int,
    seed: int,
) -> list[tuple[str, str]]:
    """Sample recordings with transcripts, up to `per_transcriber` per transcriber.

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
        for recording in recordings[:per_transcriber]:
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
    aligner = MmsAligner(device=device, star_edges=not args.no_star)

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
        groups = build_pseudo_spans(spans, args.min_seconds, args.max_seconds)
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
        "--no-star",
        action="store_true",
        help="do not wrap the transcript in MMS_FA's '*' wildcard token",
    )
    validate.add_argument("--seed", type=int, default=1)
    validate.set_defaults(func=run_validate)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
