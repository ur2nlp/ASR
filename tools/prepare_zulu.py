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

Usage:
    python tools/prepare_zulu.py
    python tools/prepare_zulu.py --input-dir data/zulu/raw \
        --output-dir data/zulu/prepared
"""

import argparse
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


def prepare_recording(
    audio_path: str,
    transcript_path: str,
    output_dir: str,
    min_seconds: float,
) -> int:
    """Segment one recording into cleaned utterance pairs on disk.

    Args:
        audio_path: Path to the source ``.wav`` recording.
        transcript_path: Path to the matching timestamped transcript.
        output_dir: Directory to write ``{stem}_{index}.wav``/``.txt`` pairs to.
        min_seconds: Drop segments shorter than this many seconds.

    Returns:
        The number of utterance pairs written for this recording.
    """
    stem = os.path.splitext(os.path.basename(audio_path))[0]
    segments = parse_segments(transcript_path)
    if not segments:
        print(f"Warning: no speech segments in {transcript_path}", file=sys.stderr)
        return 0

    # waveform: (num_samples,) at 16 kHz mono
    waveform, _ = librosa.load(audio_path, sr=TARGET_SAMPLING_RATE, mono=True)

    written = 0
    for index, (start, end, raw_text) in enumerate(segments):
        text = clean_text(raw_text)
        if not text:
            continue
        if end - start < min_seconds:
            continue

        start_sample = int(start * TARGET_SAMPLING_RATE)
        end_sample = int(end * TARGET_SAMPLING_RATE)
        clip = waveform[start_sample:end_sample]
        if clip.size == 0:
            continue

        segment_stem = f"{stem}_{index:03d}"
        sf.write(os.path.join(output_dir, f"{segment_stem}.wav"), clip, TARGET_SAMPLING_RATE)
        with open(
            os.path.join(output_dir, f"{segment_stem}.txt"), "w", encoding="utf-8"
        ) as text_file:
            text_file.write(text + "\n")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    audio_files = sorted(
        name for name in os.listdir(args.input_dir) if name.endswith(".wav")
    )
    if not audio_files:
        print(f"No .wav files found in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

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
