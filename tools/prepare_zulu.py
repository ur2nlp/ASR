"""Segment long-form Zulu recordings into short utterance pairs for fine-tuning.

The raw Zulu corpus stores each 5-minute recording and its timestamped
transcript as a sibling ``{stem}.wav`` / ``{stem}.txt`` pair in a single
directory (``data/zulu/raw/``). Each transcript is a sequence of timestamp
markers and content lines::

    [0.000]
    <no-speech>
    [4.180]
    Iyiphi i consult yakho yamaphupho ...
    [26.060]
    <no-speech>
    ...

A timestamp ``[t]`` marks the start of the span that runs until the next
timestamp; the content between two timestamps is that span's transcript.
``<no-speech>`` spans are silence and are dropped.

This script slices each recording at its timestamps into short utterance-level
``.wav``/``.txt`` pairs (resampled to 16 kHz mono) written side by side into a
single output directory, which is exactly the layout the ``paired`` dataset
loader expects. Text cleaning removes structural/annotation markup only —
``[timestamps]``, ``<...>`` tokens, and ``(...)`` parentheticals such as
``(ahleke)`` (laughter) — while keeping code-switched English verbatim. Ordinary
lowercasing and punctuation stripping are left to the training pipeline's
``preprocessing`` stage.

Spans longer than 30 s are lost to Whisper's window. With ``--long-span-manifest``
(``manifest.csv`` from ``tools/align_long_spans.py align``), each span the manifest covers
is written as its forced-alignment pieces instead of whole: piece ``k`` of span ``i``
becomes ``{stem}_{i:03d}_{k:02d}``, and only pieces with status ``ok`` are kept. Short
spans are written exactly as without the manifest. The manifest's text and times are
checked against the transcript, and a mismatch (a stale manifest) is an error. Write the
result to a new directory (``data/zulu/prepared_v2``, used by ``dataset=zulu_v2``), not over
``prepared/``: every cache and split built on the old corpus would otherwise go stale.

Usage:
    python tools/prepare_zulu.py
    python tools/prepare_zulu.py --input-dir data/zulu/raw \
        --output-dir data/zulu/prepared
    python tools/prepare_zulu.py --output-dir data/zulu/prepared_v2 \
        --long-span-manifest outputs/long_span_alignment/manifest.csv
"""

import argparse
import csv
import os
import re
import sys

import librosa
import soundfile as sf

TARGET_SAMPLING_RATE = 16000

TIMESTAMP_LINE = re.compile(r"^\[(\d+(?:\.\d+)?)\]\s*$")
ANNOTATION = re.compile(r"\([^)]*\)|<[^>]*>|\[[^\]]*\]")
WHITESPACE = re.compile(r"\s+")


def parse_segments(transcript_path: str) -> list[tuple[float, float, str]]:
    """Parse a timestamped transcript into (start, end, text) spans.

    Args:
        transcript_path: Path to the timestamped transcript file.

    Returns:
        A list of spans, each a (start_seconds, end_seconds, raw_text) tuple in
        file order. ``<no-speech>`` and empty spans are excluded.
    """
    times: list[float] = []
    contents: list[str] = []
    current: list[str] = []

    with open(transcript_path, encoding="utf-8") as transcript_file:
        for line in transcript_file:
            match = TIMESTAMP_LINE.match(line.strip())
            if match is not None:
                if times:
                    contents.append(" ".join(current).strip())
                current = []
                times.append(float(match.group(1)))
            else:
                stripped = line.strip()
                if stripped:
                    current.append(stripped)
    if times:
        contents.append(" ".join(current).strip())

    segments: list[tuple[float, float, str]] = []
    for index in range(len(times) - 1):
        start = times[index]
        end = times[index + 1]
        text = contents[index]
        if not text or text.lower() == "<no-speech>":
            continue
        segments.append((start, end, text))
    return segments


def clean_text(text: str) -> str:
    """Strip structural/annotation markup, keeping code-switched words verbatim.

    Removes ``(...)`` parentheticals, ``<...>`` tokens, and ``[...]`` markers,
    then collapses whitespace. Lowercasing and punctuation removal are handled
    downstream by the training pipeline.

    Args:
        text: Raw span text.

    Returns:
        The cleaned transcript, which may be empty if the span held only markup.
    """
    without_markup = ANNOTATION.sub(" ", text)
    return WHITESPACE.sub(" ", without_markup).strip()


def load_long_span_manifest(manifest_path: str) -> dict[str, dict[int, list[dict]]]:
    """Read an ``align_long_spans`` manifest, grouped by recording and span.

    Args:
        manifest_path: Path to ``manifest.csv``.

    Returns:
        ``{recording: {span_index: [piece rows in piece order]}}``, each row with
        ``piece`` (int), ``start``/``end`` (float), ``status`` and ``text``.
    """
    manifest: dict[str, dict[int, list[dict]]] = {}
    with open(manifest_path, encoding="utf-8", newline="") as manifest_file:
        for row in csv.DictReader(manifest_file):
            piece = {
                "piece": int(row["piece"]),
                "start": float(row["start"]),
                "end": float(row["end"]),
                "status": row["status"],
                "text": row["text"],
            }
            spans = manifest.setdefault(row["recording"], {})
            spans.setdefault(int(row["span_index"]), []).append(piece)
    for spans in manifest.values():
        for pieces in spans.values():
            pieces.sort(key=lambda piece: piece["piece"])
    return manifest


def check_pieces(
    pieces: list[dict],
    start: float,
    end: float,
    text: str,
    where: str,
) -> None:
    """Raise if a span's manifest pieces do not reproduce the span from the transcript.

    The pieces must start at the span's start, end at its end, abut each other, and their
    texts joined with spaces must equal the span's cleaned text.
    """
    tolerance = 1e-3
    if abs(pieces[0]["start"] - start) > tolerance or abs(pieces[-1]["end"] - end) > tolerance:
        raise ValueError(
            f"{where}: manifest pieces cover [{pieces[0]['start']}, {pieces[-1]['end']}], "
            f"transcript span is [{start}, {end}] -- stale manifest?"
        )
    for previous, following in zip(pieces, pieces[1:]):
        if abs(previous["end"] - following["start"]) > tolerance:
            raise ValueError(
                f"{where}: manifest pieces {previous['piece']} and {following['piece']} do not abut"
            )
    joined = " ".join(piece["text"] for piece in pieces)
    if joined != text:
        raise ValueError(f"{where}: manifest text differs from the transcript -- stale manifest?")


def write_pair(
    waveform,
    start: float,
    end: float,
    text: str,
    output_dir: str,
    segment_stem: str,
) -> bool:
    """Write one clip and its transcript; return False if the clip is empty."""
    start_sample = int(start * TARGET_SAMPLING_RATE)
    end_sample = int(end * TARGET_SAMPLING_RATE)
    clip = waveform[start_sample:end_sample]
    if clip.size == 0:
        return False
    sf.write(os.path.join(output_dir, f"{segment_stem}.wav"), clip, TARGET_SAMPLING_RATE)
    with open(os.path.join(output_dir, f"{segment_stem}.txt"), "w", encoding="utf-8") as text_file:
        text_file.write(text + "\n")
    return True


def prepare_recording(
    audio_path: str,
    transcript_path: str,
    output_dir: str,
    min_seconds: float,
    long_span_pieces: dict[int, list[dict]] | None = None,
) -> int:
    """Segment one recording into cleaned utterance pairs on disk.

    Args:
        audio_path: Path to the source ``.wav`` recording.
        transcript_path: Path to the matching timestamped transcript.
        output_dir: Directory to write ``{stem}_{index}.wav``/``.txt`` pairs to.
        min_seconds: Drop segments (and long-span pieces) shorter than this many seconds.
        long_span_pieces: This recording's spans from a long-span manifest, keyed by span
            index. Each listed span is written as its ``ok`` pieces,
            ``{stem}_{index:03d}_{piece:02d}``, instead of whole.

    Returns:
        The number of utterance pairs written for this recording.
    """
    stem = os.path.splitext(os.path.basename(audio_path))[0]
    segments = parse_segments(transcript_path)
    if not segments:
        print(f"Warning: no speech segments in {transcript_path}", file=sys.stderr)
        return 0

    # check the whole manifest for this recording before writing any of its files
    long_span_pieces = long_span_pieces or {}
    unknown = set(long_span_pieces) - set(range(len(segments)))
    if unknown:
        raise ValueError(f"{stem}: manifest lists span(s) {sorted(unknown)} not in the transcript")
    for index, pieces in long_span_pieces.items():
        start, end, raw_text = segments[index]
        check_pieces(pieces, start, end, clean_text(raw_text), where=f"{stem} span {index}")

    # waveform: (num_samples,) at 16 kHz mono
    waveform, _ = librosa.load(audio_path, sr=TARGET_SAMPLING_RATE, mono=True)

    written = 0
    for index, (start, end, raw_text) in enumerate(segments):
        text = clean_text(raw_text)
        if not text:
            continue

        pieces = long_span_pieces.get(index)
        if pieces is not None:
            for piece in pieces:
                if piece["status"] != "ok":
                    continue
                if piece["end"] - piece["start"] < min_seconds:
                    continue
                piece_stem = f"{stem}_{index:03d}_{piece['piece']:02d}"
                if write_pair(
                    waveform,
                    piece["start"],
                    piece["end"],
                    piece["text"],
                    output_dir,
                    piece_stem,
                ):
                    written += 1
            continue

        if end - start < min_seconds:
            continue
        if write_pair(waveform, start, end, text, output_dir, f"{stem}_{index:03d}"):
            written += 1

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment long-form Zulu recordings into utterance pairs",
    )
    parser.add_argument(
        "--input-dir",
        default="data/zulu/raw",
        help="Directory of sibling {stem}.wav / {stem}.txt recording pairs",
    )
    parser.add_argument(
        "--output-dir",
        default="data/zulu/prepared",
        help="Directory to write segmented .wav/.txt pairs to",
    )
    parser.add_argument(
        "--min-seconds",
        type=float,
        default=0.4,
        help="Drop segments shorter than this many seconds",
    )
    parser.add_argument(
        "--long-span-manifest",
        default=None,
        help="manifest.csv from tools/align_long_spans.py align: write the spans it "
        "covers as their ok pieces instead of whole",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    audio_files = sorted(name for name in os.listdir(args.input_dir) if name.endswith(".wav"))
    if not audio_files:
        print(f"No .wav files found in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    manifest = {}
    if args.long_span_manifest is not None:
        manifest = load_long_span_manifest(args.long_span_manifest)
        print(
            f"Long-span manifest: {sum(len(spans) for spans in manifest.values())} span(s) "
            f"in {len(manifest)} recording(s)",
            file=sys.stderr,
        )

    total_pairs = 0
    total_recordings = 0
    for audio_name in audio_files:
        stem = os.path.splitext(audio_name)[0]
        audio_path = os.path.join(args.input_dir, audio_name)
        transcript_path = os.path.join(args.input_dir, f"{stem}.txt")
        if not os.path.isfile(transcript_path):
            print(f"Warning: no transcript for {audio_name}, skipping", file=sys.stderr)
            continue
        written = prepare_recording(
            audio_path,
            transcript_path,
            args.output_dir,
            args.min_seconds,
            manifest.get(stem),
        )
        total_pairs += written
        total_recordings += 1
        print(f"{audio_name}: {written} segment(s)", file=sys.stderr)

    print(
        f"Wrote {total_pairs} utterance pair(s) from {total_recordings} "
        f"recording(s) to {args.output_dir}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
