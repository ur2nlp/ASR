"""Tests for tools/split_dry_run.py."""

import csv
import json
import os
import subprocess
import sys

from omegaconf import OmegaConf

from src.splits import SplitSpec, assign_splits
from tools.split_dry_run import read_examples_csv, summarize

ROWS = [
    {"recording": "r1", "transcriber": "A"},
    {"recording": "r2", "transcriber": "A"},
    {"recording": "r3", "transcriber": "B"},
    {"recording": "r4", "transcriber": "C"},
]
NAMES = [
    f"{recording}_{index:03d}.wav" for recording in ("r1", "r2", "r3", "r4") for index in range(2)
]
# r1 holds the only over-length segment (40 s)
DURATIONS = [40.0, 10.0, 20.0, 20.0, 15.0, 15.0, 5.0, 25.0]


def write_metadata(directory) -> str:
    path = os.path.join(directory, "metadata.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ROWS[0]))
        writer.writeheader()
        writer.writerows(ROWS)
    return path


def make_spec(directory, **overrides) -> SplitSpec:
    config = {
        "enabled": True,
        "name": "dry",
        "metadata": write_metadata(directory),
        "key_pattern": r"^(.+)_\d+\.wav$",
        "dev_size": 0,
        "test": {"transcriber": ["B"]},
    }
    config.update(overrides)
    return SplitSpec.from_config(OmegaConf.create(config))


class TestSummarize:
    def test_counts_all_and_usable(self, tmp_path):
        spec = make_spec(str(tmp_path))
        assignment = assign_splits(NAMES, spec, durations=DURATIONS, max_duration=30.0)
        report = summarize(assignment, NAMES, DURATIONS, spec, max_duration=30.0, by=None)
        assert report["test"] == {
            "examples": 2,
            "hours": round(30.0 / 3600, 3),
            "usable_examples": 2,
            "usable_hours": round(30.0 / 3600, 3),
        }
        assert report["train"]["examples"] == 6
        assert report["train"]["usable_examples"] == 5
        assert report["dev"]["examples"] == 0

    def test_breaks_usable_hours_down_by_column(self, tmp_path):
        spec = make_spec(str(tmp_path))
        assignment = assign_splits(NAMES, spec, durations=DURATIONS, max_duration=30.0)
        report = summarize(assignment, NAMES, DURATIONS, spec, max_duration=30.0, by="transcriber")
        # A: 10 + 20 + 20 usable (the 40 s segment is filtered); C: 5 + 25
        assert report["train"]["usable_hours_by"] == {
            "A": round(50.0 / 3600, 3),
            "C": round(30.0 / 3600, 3),
        }

    def test_no_filter_counts_everything_usable(self, tmp_path):
        spec = make_spec(str(tmp_path))
        assignment = assign_splits(NAMES, spec, durations=DURATIONS)
        report = summarize(assignment, NAMES, DURATIONS, spec, max_duration=None, by=None)
        assert report["train"]["usable_examples"] == report["train"]["examples"]


def test_read_examples_csv_keeps_order(tmp_path):
    path = tmp_path / "examples.csv"
    path.write_text("example,duration\nb.wav,2.5\na.wav,1\n", encoding="utf-8")
    assert read_examples_csv(str(path)) == (["b.wav", "a.wav"], [2.5, 1.0])


def test_command_line_composes_split_config(tmp_path):
    """End to end through Hydra, with the repo's example split config overridden."""
    metadata = write_metadata(str(tmp_path))
    examples = tmp_path / "examples.csv"
    with open(examples, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["example", "duration"])
        writer.writerows(zip(NAMES, DURATIONS))
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.split_dry_run",
            "--examples",
            str(examples),
            "dataset=example_paired",
            "split=example_metadata",
            f"split.metadata={metadata}",
            r"split.key_pattern='^(.+)_\d+\.wav$'",
            "split.dev_size=0",
            "split.test={transcriber:[B]}",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["split_id"].startswith("example-")
    assert report["max_duration"] == 30.0
    assert report["splits"]["test"]["examples"] == 2
    assert report["splits"]["train"]["usable_examples"] == 5
