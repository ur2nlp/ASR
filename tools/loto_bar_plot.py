"""
Bar chart of test WER for the matched-pair LOTO runs, per held-out transcriber.

Each run is classified from its fetched config, not from its name:

- held-out transcriber: `split.test.transcriber`
- with / without the transcriber in training: `split.test_remainder` (`train` / `exclude`)
- vocabulary: `focus.enabled` (FOCUS / pretrained)

Runs without `split.test_remainder` (the naive LOTO folds) are skipped: they use a different
test set, so their WER is not comparable with the matched pairs. A cell with no test score
yet is drawn as a labelled placeholder instead of being silently dropped.

Each panel is labelled with the size of T's test half, and panels are ordered by it, largest
first. The segment count is what the runs actually evaluated (test runtime x samples per
second from the trainer state). Hours come from `--sizes`, a CSV of `transcriber,test_hours`;
use *usable* (<= 30 s) hours there, not the `hours` in the split's `split.yaml`, which are
counted before the length filter and overstate the short-segment-biased folds by up to ~3x.

Usage:
    python tools/loto_bar_plot.py
    python tools/loto_bar_plot.py --output plots/loto_test_wer.png --csv outputs/loto_test_wer.csv
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
import plotnine as p9
import yaml

WITH_LABEL = "with T in training"
WITHOUT_LABEL = "without T"
FILL_COLORS = {WITHOUT_LABEL: "#eb6834", WITH_LABEL: "#2a78d6"}
VOCAB_ORDER = ["pretrained", "FOCUS"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--states-dir", type=Path, default=Path("outputs/trainer_states"))
    parser.add_argument("--configs-dir", type=Path, default=Path("outputs/configs"))
    parser.add_argument(
        "--pattern",
        default=r"loto-.*",
        help="regex fullmatched against experiment ids (config file stems)",
    )
    parser.add_argument("--metric", default="test_wer")
    parser.add_argument("--output", type=Path, default=Path("plots/loto_test_wer.png"))
    parser.add_argument("--csv", type=Path, default=None, help="also write the table here")
    parser.add_argument(
        "--sizes",
        type=Path,
        default=Path("outputs/loto_test_sizes.csv"),
        help="CSV of transcriber,test_hours (usable hours of T's test half); optional",
    )
    return parser.parse_args()


def read_test_results(state_path: Path, metric: str) -> tuple[float | None, int | None]:
    """Return `metric` and the number of test examples evaluated, from a trainer state.

    The example count is not logged directly; it is recovered as test runtime times test
    samples per second, which HF rounds to 3 decimals -- well under one example of error.

    Returns:
        `(value, num_examples)`, each None when the state or the test entry is absent.
    """
    if not state_path.exists():
        return None, None
    with open(state_path) as state_file:
        state = json.load(state_file)
    test_entry = None
    for entry in state.get("log_history", []):
        if metric in entry:
            test_entry = entry
    if test_entry is None:
        return None, None
    num_examples = round(test_entry["test_runtime"] * test_entry["test_samples_per_second"])
    return test_entry[metric], num_examples


def collect_runs(
    configs_dir: Path,
    states_dir: Path,
    pattern: str,
    metric: str,
) -> pd.DataFrame:
    """Build one row per matched-pair run: transcriber, vocabulary, condition, metric."""
    rows = []
    for config_path in sorted(configs_dir.glob("*.yaml")):
        experiment_id = config_path.stem
        if not re.fullmatch(pattern, experiment_id):
            continue
        with open(config_path) as config_file:
            config = yaml.safe_load(config_file)

        split = config.get("split") or {}
        remainder = split.get("test_remainder")
        if not split.get("enabled") or remainder not in ("train", "exclude"):
            print(f"skip {experiment_id}: not a matched-pair run", file=sys.stderr)
            continue
        held_out = split["test"]["transcriber"]
        if len(held_out) != 1:
            print(f"skip {experiment_id}: holds out {held_out}", file=sys.stderr)
            continue

        state_path = states_dir / f"{experiment_id}.json"
        value, num_examples = read_test_results(state_path, metric)
        rows.append(
            {
                "experiment_id": experiment_id,
                "transcriber": held_out[0],
                "vocabulary": "FOCUS"
                if (config.get("focus") or {}).get("enabled")
                else "pretrained",
                "condition": WITH_LABEL if remainder == "train" else WITHOUT_LABEL,
                metric: value,
                "test_segments": num_examples,
                "has_state": state_path.exists(),
            }
        )
    return pd.DataFrame(rows)


def complete_grid(runs: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Add a row for every missing (transcriber, vocabulary, condition) cell.

    Only vocabularies that at least one run of a transcriber used are filled in, so a
    transcriber with FOCUS runs only does not sprout empty pretrained placeholders.
    """
    cells = []
    for transcriber, group in runs.groupby("transcriber"):
        for vocabulary in group["vocabulary"].unique():
            for condition in (WITHOUT_LABEL, WITH_LABEL):
                cells.append((transcriber, vocabulary, condition))
    grid = pd.DataFrame(cells, columns=["transcriber", "vocabulary", "condition"])
    merged = grid.merge(runs, on=["transcriber", "vocabulary", "condition"], how="left")

    # label each cell: its value, or why it has none
    labels = []
    for _, row in merged.iterrows():
        if pd.notna(row[metric]):
            labels.append(f"{100 * row[metric]:.1f}")
        elif row["has_state"] is True:
            labels.append("still\nrunning")
        else:
            labels.append("not\nrun")
    merged["label"] = labels
    return merged


def load_test_hours(sizes_path: Path) -> dict[str, float]:
    """Read `transcriber,test_hours` rows; an absent file gives no hours."""
    if not sizes_path.exists():
        print(f"no {sizes_path}; panels show segment counts only", file=sys.stderr)
        return {}
    sizes = pd.read_csv(sizes_path, dtype={"transcriber": str})
    return dict(zip(sizes["transcriber"], sizes["test_hours"]))


def test_set_sizes(cells: pd.DataFrame, test_hours: dict[str, float]) -> pd.DataFrame:
    """One row per transcriber: segments evaluated, usable hours, and the panel label.

    Every run of a transcriber shares one test set, so their segment counts must agree;
    a disagreement means the runs did not evaluate the same utterances, and is reported.
    """
    rows = []
    for transcriber, group in cells.groupby("transcriber"):
        counts = set(group["test_segments"].dropna().astype(int))
        if len(counts) > 1:
            print(
                f"warning: {transcriber} runs evaluated different test sizes {sorted(counts)}",
                file=sys.stderr,
            )
        segments = max(counts) if counts else None
        hours = test_hours.get(transcriber)

        size_parts = []
        if hours is not None:
            size_parts.append(f"{hours:.1f} h")
        if segments is not None:
            size_parts.append(f"{segments:,} segs")
        size_text = ", ".join(size_parts) if size_parts else "size unknown"
        rows.append(
            {
                "transcriber": transcriber,
                "test_hours": hours,
                "test_segments": segments,
                "panel": f"{transcriber}\nT test half: {size_text}",
            }
        )
    sizes = pd.DataFrame(rows)
    # order panels by test size, largest first: hours when known, else segments
    sizes = sizes.sort_values(
        ["test_hours", "test_segments"],
        ascending=False,
        na_position="last",
    )
    return sizes.reset_index(drop=True)


def build_plot(cells: pd.DataFrame, sizes: pd.DataFrame, metric: str) -> p9.ggplot:
    cells = cells.merge(sizes[["transcriber", "panel"]], on="transcriber", how="left")
    cells["panel"] = pd.Categorical(cells["panel"], categories=list(sizes["panel"]))
    cells["vocabulary"] = pd.Categorical(cells["vocabulary"], categories=VOCAB_ORDER)
    cells["condition"] = pd.Categorical(
        cells["condition"],
        categories=[WITHOUT_LABEL, WITH_LABEL],
    )
    # place labels above bars; placeholders sit just above the baseline
    cells["label_y"] = cells[metric].fillna(0.0) + 0.012
    # values in primary ink, placeholders in muted ink
    cells["label_color"] = cells[metric].isna().map({False: "#0b0b0b", True: "#898781"})
    cells["label_size"] = cells[metric].isna().map({False: 8.0, True: 6.5})

    dodge = p9.position_dodge(width=0.8)
    metric_name = metric.removeprefix("test_").upper()
    return (
        p9.ggplot(cells, p9.aes(x="vocabulary", y=metric, fill="condition"))
        + p9.geom_col(position=dodge, width=0.76, na_rm=True)
        + p9.geom_text(
            p9.aes(
                y="label_y",
                label="label",
                group="condition",
                color="label_color",
                size="label_size",
            ),
            position=dodge,
            va="bottom",
            lineheight=0.9,
        )
        + p9.scale_color_identity()
        + p9.scale_size_identity()
        + p9.facet_wrap("panel", nrow=1, scales="free_x")
        + p9.scale_fill_manual(values=FILL_COLORS)
        + p9.scale_y_continuous(
            labels=lambda values: [f"{100 * value:.0f}" for value in values],
            limits=(0, None),
            expand=(0, 0, 0.08, 0),
        )
        + p9.labs(
            x="",
            y=f"test {metric_name} (%)",
            title=f"Matched-pair LOTO: test {metric_name} by held-out transcriber T",
            subtitle="Panels ordered by the size of T's test half (usable <= 30 s audio)",
            caption=(
                "Within a panel all bars share one test set. Compare within a panel, "
                "not across: training sizes differ (49-86 h)\nand the 38f6f23e, a5d2e6b9, "
                "71f4256c test sets are short-segment biased (.claude/long_segments.md)."
            ),
        )
        + p9.theme_minimal(base_size=10)
        + p9.theme(
            figure_size=(11, 4.2),
            legend_position="top",
            legend_title=p9.element_blank(),
            panel_grid_major_x=p9.element_blank(),
            panel_grid_minor=p9.element_blank(),
            panel_grid_major_y=p9.element_line(color="#e4e3df", size=0.4),
            plot_caption=p9.element_text(color="#52514e", size=7.5, ha="left"),
            plot_background=p9.element_rect(fill="#fcfcfb", color="none"),
            plot_subtitle=p9.element_text(color="#52514e", size=9),
            strip_text=p9.element_text(lineheight=1.3),
        )
    )


def main() -> None:
    args = parse_args()
    runs = collect_runs(args.configs_dir, args.states_dir, args.pattern, args.metric)
    if runs.empty:
        sys.exit("no matched-pair runs found")
    cells = complete_grid(runs, args.metric)
    sizes = test_set_sizes(cells, load_test_hours(args.sizes))
    print(sizes[["transcriber", "test_hours", "test_segments"]].to_string(index=False))

    table = cells[["transcriber", "vocabulary", "condition", "experiment_id", args.metric]]
    table = table.sort_values(["transcriber", "vocabulary", "condition"])
    print(table.to_string(index=False))
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.csv, index=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    build_plot(cells, sizes, args.metric).save(args.output, dpi=200, verbose=False)
    print(f"wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
