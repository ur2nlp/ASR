"""Tests for tools/prepare_zulu.py, in particular the long-span manifest path."""

import csv
import os

import numpy as np
import pytest
import soundfile as sf

from tools.prepare_zulu import TARGET_SAMPLING_RATE, prepare_recording

STEM = "0" * 32
TRANSCRIPT = """[0.000]
<no-speech>
[1.000]
Sawubona baba
[5.000]
<no-speech>
[5.500]
Ngiyaphila (ahleke) ngiyabonga kakhulu wena
[45.000]
<no-speech>
[46.000]
"""
# after <no-speech> spans are dropped: index 0 = [1, 5], index 1 = [5.5, 45] (40 s)


@pytest.fixture
def recording(tmp_path):
    audio_path = tmp_path / f"{STEM}.wav"
    transcript_path = tmp_path / f"{STEM}.txt"
    samples = np.random.default_rng(1).uniform(-0.1, 0.1, 46 * TARGET_SAMPLING_RATE)
    sf.write(audio_path, samples.astype(np.float32), TARGET_SAMPLING_RATE)
    transcript_path.write_text(TRANSCRIPT, encoding="utf-8")
    output_dir = tmp_path / "prepared"
    output_dir.mkdir()
    return str(audio_path), str(transcript_path), output_dir


def pieces(*rows):
    return [
        {"piece": index, "start": start, "end": end, "status": status, "text": text}
        for index, (start, end, status, text) in enumerate(rows)
    ]


def written(output_dir) -> dict[str, tuple[float, str]]:
    """Map each written stem to its (duration in seconds, text)."""
    result = {}
    for name in sorted(os.listdir(output_dir)):
        if name.endswith(".wav"):
            stem = name[:-4]
            info = sf.info(os.path.join(output_dir, name))
            text = (output_dir / f"{stem}.txt").read_text(encoding="utf-8").strip()
            result[stem] = (round(info.duration, 2), text)
    return result


class TestWithoutManifest:
    def test_long_span_is_written_whole(self, recording):
        audio_path, transcript_path, output_dir = recording
        count = prepare_recording(audio_path, transcript_path, str(output_dir), 0.4)
        assert count == 2
        assert written(output_dir) == {
            f"{STEM}_000": (4.0, "Sawubona baba"),
            f"{STEM}_001": (39.5, "Ngiyaphila ngiyabonga kakhulu wena"),
        }


class TestWithManifest:
    def test_long_span_is_replaced_by_its_pieces(self, recording):
        audio_path, transcript_path, output_dir = recording
        manifest = {
            1: pieces(
                (5.5, 20.0, "ok", "Ngiyaphila ngiyabonga"),
                (20.0, 45.0, "ok", "kakhulu wena"),
            )
        }
        count = prepare_recording(audio_path, transcript_path, str(output_dir), 0.4, manifest)
        assert count == 3
        assert written(output_dir) == {
            f"{STEM}_000": (4.0, "Sawubona baba"),
            f"{STEM}_001_00": (14.5, "Ngiyaphila ngiyabonga"),
            f"{STEM}_001_01": (25.0, "kakhulu wena"),
        }

    def test_pieces_not_ok_are_skipped(self, recording):
        audio_path, transcript_path, output_dir = recording
        manifest = {
            1: pieces(
                (5.5, 20.0, "ok", "Ngiyaphila ngiyabonga"),
                (20.0, 45.0, "too_long", "kakhulu wena"),
            )
        }
        prepare_recording(audio_path, transcript_path, str(output_dir), 0.4, manifest)
        assert set(written(output_dir)) == {f"{STEM}_000", f"{STEM}_001_00"}

    def test_stale_text_raises_before_writing_anything(self, recording):
        audio_path, transcript_path, output_dir = recording
        manifest = {
            1: pieces(
                (5.5, 20.0, "ok", "Ngiyaphila"),
                (20.0, 45.0, "ok", "kakhulu wena"),
            )
        }
        with pytest.raises(ValueError, match="text differs"):
            prepare_recording(audio_path, transcript_path, str(output_dir), 0.4, manifest)
        assert os.listdir(output_dir) == []

    def test_stale_times_raise(self, recording):
        audio_path, transcript_path, output_dir = recording
        manifest = {1: pieces((6.0, 45.0, "ok", "Ngiyaphila ngiyabonga kakhulu wena"))}
        with pytest.raises(ValueError, match="stale manifest"):
            prepare_recording(audio_path, transcript_path, str(output_dir), 0.4, manifest)

    def test_unknown_span_raises(self, recording):
        audio_path, transcript_path, output_dir = recording
        manifest = {7: pieces((0.0, 1.0, "ok", "x"))}
        with pytest.raises(ValueError, match="not in the transcript"):
            prepare_recording(audio_path, transcript_path, str(output_dir), 0.4, manifest)


def test_manifest_round_trip(tmp_path):
    from tools.prepare_zulu import load_long_span_manifest

    path = tmp_path / "manifest.csv"
    with open(path, "w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=["recording", "span_index", "piece", "start", "end", "status", "text"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "recording": STEM,
                "span_index": 1,
                "piece": 1,
                "start": 20.0,
                "end": 45.0,
                "status": "ok",
                "text": "kakhulu wena",
            }
        )
        writer.writerow(
            {
                "recording": STEM,
                "span_index": 1,
                "piece": 0,
                "start": 5.5,
                "end": 20.0,
                "status": "ok",
                "text": "Ngiyaphila ngiyabonga",
            }
        )
    manifest = load_long_span_manifest(str(path))
    assert [piece["piece"] for piece in manifest[STEM][1]] == [0, 1]
    assert manifest[STEM][1][0]["start"] == 5.5


def test_plan_matches_what_is_written(recording):
    """--list-only lists exactly the pairs a real run writes, with timestamp durations."""
    from tools.prepare_zulu import plan_recording

    audio_path, transcript_path, output_dir = recording
    manifest = {
        1: pieces(
            (5.5, 20.0, "ok", "Ngiyaphila ngiyabonga"),
            (20.0, 45.0, "low_score", "kakhulu wena"),
        )
    }
    planned = plan_recording(STEM, transcript_path, 0.4, manifest)
    prepare_recording(audio_path, transcript_path, str(output_dir), 0.4, manifest)
    files = written(output_dir)
    assert [name for name, *_ in planned] == list(files)
    for name, start, end, text in planned:
        assert files[name] == (round(end - start, 2), text)
