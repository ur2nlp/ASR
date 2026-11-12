"""Dry-run a metadata split: what train/dev/test would hold, before any features exist.

Runs the same `splits.assign_splits` call as training (`src/__main__.py:_assign_splits`),
on a list of examples and their durations instead of the feature cache, and reports each
split's size: all examples, and the *usable* ones that survive the audio-length filter
(`dataset.max_audio_length_seconds`). With ``--by COLUMN`` it also breaks usable hours down
by a metadata column (e.g. transcriber). Use it to check a split -- in particular a
``train_budget: matched`` one, whose hours depend on durations -- before launching runs.

Examples come from either:

- ``--examples CSV`` with columns ``example`` (the key source: the audio file name when
  ``split.key_from`` is ``path``) and ``duration`` (seconds), e.g. listed by a data
  preparation script before any audio is written; or
- ``--audio-dir DIR``: every ``--audio-ext`` file in DIR (not recursive), with durations
  read from the file headers.

Caveat: training assigns over the feature cache in *its* row order, and durations there are
decoded sample counts. If the assignment's sampling is order-sensitive, a dry run in a
different order can pick different groups; sizes still come out comparable.

The configuration is composed with Hydra exactly as for training, so pass the same
overrides. The report goes to stdout as JSON (``--output`` also writes it to a file); a
readable table goes to stderr.

Usage:
    python tools/split_dry_run.py --examples examples.csv --by transcriber \\
        dataset=<lang> split=<split> 'split.test={transcriber:[T1]}' split.test_remainder=train
"""

import argparse
import csv
import json
import os
import sys

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from src import splits

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")


def read_examples_csv(path: str) -> tuple[list[str], list[float]]:
    """Read `example,duration` rows, in file order."""
    names = []
    durations = []
    with open(path, newline="", encoding="utf-8") as examples_file:
        for row in csv.DictReader(examples_file):
            names.append(row["example"])
            durations.append(float(row["duration"]))
    return names, durations


def read_audio_dir(directory: str, extension: str) -> tuple[list[str], list[float]]:
    """List audio files (sorted by name) and read their durations from the headers."""
    import soundfile

    names = sorted(name for name in os.listdir(directory) if name.endswith(extension))
    durations = [soundfile.info(os.path.join(directory, name)).duration for name in names]
    return names, durations


def compose_config(overrides: list[str]) -> DictConfig:
    """Compose `configs/main.yaml` with Hydra overrides, as training does."""
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="main", overrides=overrides)


def summarize(
    assignment: dict[str, list[int]],
    key_sources: list[str],
    durations: list[float],
    spec: splits.SplitSpec,
    max_duration: float | None,
    by: str | None,
) -> dict:
    """Count examples and hours per split, all and usable, optionally by a metadata column.

    Args:
        assignment: `{split: example indices}` from `assign_splits`.
        key_sources: The examples' key sources, in the order the indices refer to.
        durations: Each example's duration in seconds.
        spec: The split specification (for the metadata and key pattern).
        max_duration: The audio-length filter; None counts every example as usable.
        by: A metadata column to break usable hours down by, or None. Values of a
            multi-valued column are joined with the separator.

    Returns:
        `{split: {examples, hours, usable_examples, usable_hours[, usable_hours_by]}}`.
    """
    attribute_of = None
    if by is not None:
        metadata = splits.load_metadata(spec)
        keys = splits.keys_from_values(
            key_sources,
            spec.key_pattern,
            from_paths=spec.key_from == "path",
        )

        def attribute_of(index: int) -> str:
            return spec.multi_value_separator.join(metadata[keys[index]][by])

    report = {}
    for name in splits.SPLIT_NAMES:
        indices = assignment.get(name, [])
        usable = [
            index for index in indices if max_duration is None or durations[index] <= max_duration
        ]
        entry = {
            "examples": len(indices),
            "hours": round(sum(durations[index] for index in indices) / 3600.0, 3),
            "usable_examples": len(usable),
            "usable_hours": round(sum(durations[index] for index in usable) / 3600.0, 3),
        }
        if attribute_of is not None:
            seconds_by: dict[str, float] = {}
            for index in usable:
                value = attribute_of(index)
                seconds_by[value] = seconds_by.get(value, 0.0) + durations[index]
            entry["usable_hours_by"] = {
                value: round(seconds / 3600.0, 3) for value, seconds in sorted(seconds_by.items())
            }
        report[name] = entry
    return report


def print_table(report: dict) -> None:
    """Write a readable summary of the report to stderr."""
    print(f"split id: {report['split_id']}", file=sys.stderr)
    for name, entry in report["splits"].items():
        print(
            f"  {name:5s} {entry['usable_examples']:7d} usable ({entry['usable_hours']:7.2f} h) "
            f"of {entry['examples']:7d} ({entry['hours']:7.2f} h)",
            file=sys.stderr,
        )
        for value, hours in entry.get("usable_hours_by", {}).items():
            print(f"        {value}: {hours:.2f} h", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dry-run a metadata split over a list of examples and durations.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--examples", help="CSV with `example` and `duration` columns")
    source.add_argument("--audio-dir", help="directory of audio files; durations from headers")
    parser.add_argument("--audio-ext", default=".wav")
    parser.add_argument("--by", default=None, help="metadata column to break usable hours by")
    parser.add_argument("--output", default=None, help="also write the JSON report here")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, as for training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = compose_config(args.overrides)
    if not splits.is_enabled(config):
        sys.exit("no metadata split is enabled in this configuration (set split=...)")
    spec = splits.SplitSpec.from_config(config.split)

    if args.examples is not None:
        key_sources, durations = read_examples_csv(args.examples)
    else:
        key_sources, durations = read_audio_dir(args.audio_dir, args.audio_ext)
    max_duration = config.dataset.get("max_audio_length_seconds")

    assignment = splits.assign_splits(
        key_sources,
        spec,
        seed=config.seed,
        durations=durations,
        max_duration=max_duration,
    )
    report = {
        "split_id": spec.split_id(config.seed),
        "overrides": args.overrides,
        "max_duration": max_duration,
        "splits": summarize(assignment, key_sources, durations, spec, max_duration, args.by),
    }
    print_table(report)
    print(json.dumps(report, indent=2))
    if args.output is not None:
        with open(args.output, "w", encoding="utf-8") as output_file:
            json.dump(report, output_file, indent=2)


if __name__ == "__main__":
    main()
