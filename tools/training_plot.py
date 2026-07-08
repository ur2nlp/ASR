"""Plot training metrics from HuggingFace Trainer logs.

Reads `trainer_state.json` files (written into every checkpoint and the
best-checkpoint directory) and plots any logged metric over training. For CTC
ASR runs the metrics of interest are typically `loss`, `eval_loss`, `eval_wer`,
and `eval_cer`.

Usage:
    # Single run: WER over training
    python tools/training_plot.py --metric eval_wer \\
        --state-file models/zulu/xlsr300m_l40_basic/best-checkpoint/trainer_state.json

    # Compare several runs (regex matched against file paths under cwd)
    python tools/training_plot.py --metric eval_wer \\
        --state-pattern "models/.*/trainer_state\\.json"

    # Several metrics as subplots
    python tools/training_plot.py --metrics loss eval_loss eval_wer eval_cer \\
        --state-file path/to/trainer_state.json

    # Regex metric patterns (matched with re.fullmatch)
    python tools/training_plot.py --metric "eval_.*" --state-file path/to/trainer_state.json

    # With external eval sets, per-set metrics are named eval_{name}_wer
    python tools/training_plot.py --metric "eval_.*_wer" --state-file path/to/trainer_state.json

    # Save to file instead of showing
    python tools/training_plot.py --metric eval_wer --state-file path/to/trainer_state.json --output wer.png

    # Custom legend names, in order of matched files
    python tools/training_plot.py --metric eval_wer \\
        --state-pattern "models/.*/trainer_state\\.json" --run-names "baseline" "lr1e-5"

    # Set y-axis limits
    python tools/training_plot.py --metric eval_wer --state-file path/to/trainer_state.json --ylim 0 1
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
from plotnine import (
    aes,
    element_rect,
    element_text,
    facet_wrap,
    geom_line,
    geom_point,
    ggplot,
    labs,
    theme,
    theme_bw,
    theme_minimal,
    ylim,
)


def load_from_trainer_state(filepath):
    """Load log history from a trainer_state.json file."""
    with open(filepath, "r") as f:
        state = json.load(f)
    return pd.DataFrame(state["log_history"])


def find_files_by_regex(pattern, root="."):
    """Walk the directory tree and return files whose paths match the regex."""
    compiled = re.compile(pattern)
    matches = []
    for filepath in Path(root).rglob("*"):
        if filepath.is_file() and compiled.search(str(filepath)):
            matches.append(str(filepath))
    return sorted(matches)


def load_data(
    state_file=None,
    state_pattern=None,
    run_names=None,
    exclude_pattern=None,
):
    """Load training data from one or more trainer_state.json files."""
    dataframes = []

    exclude_re = re.compile(exclude_pattern) if exclude_pattern else None

    if state_pattern:
        files = find_files_by_regex(state_pattern)
        if exclude_re:
            files = [f for f in files if not exclude_re.search(f)]
        if not files:
            print(f"Warning: No files found matching pattern: {state_pattern}", file=sys.stderr)
        for idx, filepath in enumerate(files):
            df = load_from_trainer_state(filepath)
            if run_names and idx < len(run_names):
                df["run"] = run_names[idx]
            else:
                df["run"] = filepath
            dataframes.append(df)

    elif state_file:
        df = load_from_trainer_state(state_file)
        df["run"] = run_names[0] if run_names else state_file
        dataframes.append(df)

    else:
        raise ValueError("Must provide one of: --state-file or --state-pattern")

    if not dataframes:
        raise ValueError("No data loaded. Check file paths.")

    return pd.concat(dataframes, ignore_index=True)


def plot_metric(data, metric, x_axis="step", output=None, title=None, y_limits=None):
    """Create a plot for a single metric.

    Args:
        y_limits: Tuple of (lower, upper) for the y-axis. Either can be None.
    """
    metric_data = data[data[metric].notna()].copy()

    if len(metric_data) == 0:
        print(f"Warning: No data found for metric '{metric}'", file=sys.stderr)
        print(
            f"Available metrics: {[col for col in data.columns if data[col].notna().any()]}",
            file=sys.stderr,
        )
        return None

    multiple_runs = len(metric_data["run"].unique()) > 1

    # prefer 'step' on the x-axis, fall back to 'epoch'
    if x_axis not in metric_data.columns or metric_data[x_axis].isna().all():
        x_axis = "epoch" if "epoch" in metric_data.columns else "step"

    # mark the min and max points
    min_idx = metric_data[metric].idxmin()
    max_idx = metric_data[metric].idxmax()
    extrema_data = metric_data.loc[[min_idx, max_idx]].copy()

    plot = (
        ggplot(metric_data, aes(x=x_axis, y=metric))
        + (geom_line(aes(color="run"), size=1.2) if multiple_runs else geom_line(size=1.2))
        + (
            geom_point(aes(color="run"), data=extrema_data, size=4, shape="x")
            if multiple_runs
            else geom_point(data=extrema_data, size=4, shape="x")
        )
        + labs(title=title or f"{metric} over training", x=x_axis.capitalize(), y=metric)
        + theme_bw()
        + theme(
            legend_position="bottom" if multiple_runs else "none",
            axis_title=element_text(size=14),
            legend_title=element_text(size=12),
            legend_text=element_text(size=10),
            axis_text=element_text(size=10),
            figure_size=(10, 6),
            plot_background=element_rect(fill="white"),
            panel_background=element_rect(fill="white"),
        )
    )

    if y_limits:
        plot = plot + ylim(y_limits)

    if output:
        plot.save(output, dpi=300, verbose=False, transparent=False)
        print(f"Saved plot to {output}")
    else:
        plot.show()

    return plot


def expand_metric_patterns(patterns, available_metrics):
    """Expand regex patterns to matching metric names.

    Each pattern is matched with re.fullmatch against available metrics. Literal
    metric names work as-is (they are valid regexes that match themselves).

    Args:
        patterns: List of metric names or regex patterns.
        available_metrics: List of all available metric names.

    Returns:
        List of matched metric names (preserving order, no duplicates).
    """
    expanded = []
    seen = set()

    for pattern in patterns:
        try:
            compiled = re.compile(pattern)
        except re.error as error:
            print(f"Invalid regex pattern '{pattern}': {error}", file=sys.stderr)
            sys.exit(1)
        for metric in sorted(available_metrics):
            if compiled.fullmatch(metric) and metric not in seen:
                expanded.append(metric)
                seen.add(metric)

    return expanded


def print_metric_summary(data, metrics, x_axis="step"):
    """Print min/max values (and where they occurred) for each metric."""
    print("\n" + "=" * 80)
    print("METRIC SUMMARY")
    print("=" * 80)

    for metric in metrics:
        metric_data = data[data[metric].notna()].copy()
        if len(metric_data) == 0:
            continue

        print(f"\n{metric}:")
        print("-" * 80)

        min_idx = metric_data[metric].idxmin()
        max_idx = metric_data[metric].idxmax()

        min_val = metric_data.loc[min_idx, metric]
        max_val = metric_data.loc[max_idx, metric]
        min_step = metric_data.loc[min_idx, x_axis] if x_axis in metric_data.columns else "N/A"
        max_step = metric_data.loc[max_idx, x_axis] if x_axis in metric_data.columns else "N/A"
        min_run = metric_data.loc[min_idx, "run"]
        max_run = metric_data.loc[max_idx, "run"]

        print(f"  Min: {min_val:.6f} at {x_axis}={min_step} (run: {min_run})")
        print(f"  Max: {max_val:.6f} at {x_axis}={max_step} (run: {max_run})")

    print("\n" + "=" * 80 + "\n")


def plot_multiple_metrics(data, metrics, x_axis="step", output=None, y_limits=None):
    """Create faceted subplots for multiple metrics.

    Args:
        y_limits: Tuple of (lower, upper) for the y-axis. Either can be None.
    """
    plot_data = []
    extrema_data = []

    for metric in metrics:
        metric_data = data[data[metric].notna()].copy()
        metric_data["metric_name"] = metric
        metric_data["metric_value"] = metric_data[metric]
        plot_data.append(metric_data[[x_axis, "run", "metric_name", "metric_value"]])

        min_idx = metric_data[metric].idxmin()
        max_idx = metric_data[metric].idxmax()
        extrema = metric_data.loc[[min_idx, max_idx]].copy()
        extrema["metric_name"] = metric
        extrema["metric_value"] = extrema[metric]
        extrema_data.append(extrema[[x_axis, "run", "metric_name", "metric_value"]])

    plot_data = pd.concat(plot_data, ignore_index=True)
    extrema_data = pd.concat(extrema_data, ignore_index=True)

    multiple_runs = len(plot_data["run"].unique()) > 1

    plot = (
        ggplot(plot_data, aes(x=x_axis, y="metric_value"))
        + (geom_line(aes(color="run"), size=1.0) if multiple_runs else geom_line(size=1.0))
        + (
            geom_point(aes(color="run"), data=extrema_data, size=3, shape="x")
            if multiple_runs
            else geom_point(data=extrema_data, size=3, shape="x")
        )
        + facet_wrap("~metric_name", scales="free_y", ncol=2)
        + labs(x=x_axis.capitalize(), y="Value")
        + theme_minimal()
        + theme(
            legend_position="bottom" if multiple_runs else "none",
            axis_title=element_text(size=12),
            legend_title=element_text(size=10),
            legend_text=element_text(size=8),
            axis_text=element_text(size=8),
            figure_size=(12, 4 * ((len(metrics) + 1) // 2)),
            plot_background=element_rect(fill="white"),
            panel_background=element_rect(fill="white"),
        )
    )

    if y_limits:
        plot = plot + ylim(y_limits)

    if output:
        plot.save(output, dpi=300, verbose=False, transparent=False)
        print(f"Saved plot to {output}")
    else:
        plot.show()

    return plot


def main():
    parser = argparse.ArgumentParser(
        description="Plot training metrics from HuggingFace Trainer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--state-file", type=str, help="Path to a trainer_state.json file")
    parser.add_argument(
        "--state-pattern",
        type=str,
        help="Regex pattern for multiple trainer_state.json files (searched from cwd)",
    )
    parser.add_argument(
        "--exclude-pattern",
        type=str,
        help="Regex pattern to exclude matched files",
    )

    parser.add_argument(
        "--metric",
        type=str,
        help="Single metric or regex pattern (e.g. eval_wer, eval_.*loss)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        help="Multiple metrics or regex patterns to plot as subplots",
    )

    parser.add_argument(
        "--x-axis",
        type=str,
        default="step",
        choices=["step", "epoch"],
        help="X-axis variable (default: step)",
    )
    parser.add_argument("--output", type=str, help="Save plot to file instead of showing")
    parser.add_argument("--title", type=str, help="Custom plot title")
    parser.add_argument(
        "--run-names",
        nargs="+",
        help="Custom names for runs (in order of matched files)",
    )
    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="List available metrics and exit",
    )
    parser.add_argument(
        "--ylim",
        nargs="+",
        type=float,
        metavar="VALUE",
        help="Y-axis limits: one value for lower bound, two for (lower, upper)",
    )

    args = parser.parse_args()

    if not any([args.state_file, args.state_pattern]):
        parser.error("Must provide one of: --state-file or --state-pattern")

    if not args.list_metrics and not args.metric and not args.metrics:
        parser.error("Must provide either --metric or --metrics (or use --list-metrics)")

    y_limits = None
    if args.ylim:
        if len(args.ylim) == 1:
            y_limits = (args.ylim[0], None)
        elif len(args.ylim) == 2:
            y_limits = (args.ylim[0], args.ylim[1])
        else:
            parser.error("--ylim accepts 1 or 2 values")

    data = load_data(
        state_file=args.state_file,
        state_pattern=args.state_pattern,
        run_names=args.run_names,
        exclude_pattern=args.exclude_pattern,
    )

    available_metrics = [
        col
        for col in data.columns
        if data[col].notna().any() and col not in ["run", "step", "epoch"]
    ]

    if args.list_metrics:
        print("Available metrics:")
        for metric in sorted(available_metrics):
            count = data[metric].notna().sum()
            print(f"  {metric} ({count} values)")
        return

    patterns = args.metrics if args.metrics else [args.metric]
    metrics = expand_metric_patterns(patterns, available_metrics)

    if not metrics:
        print(f"Error: No metrics matched the pattern(s): {patterns}", file=sys.stderr)
        print(f"Available metrics: {sorted(available_metrics)}", file=sys.stderr)
        sys.exit(1)

    print(f"Plotting {len(metrics)} metric(s): {', '.join(metrics)}", file=sys.stderr)

    if len(metrics) == 1:
        plot_metric(
            data,
            metrics[0],
            x_axis=args.x_axis,
            output=args.output,
            title=args.title,
            y_limits=y_limits,
        )
    else:
        plot_multiple_metrics(
            data,
            metrics,
            x_axis=args.x_axis,
            output=args.output,
            y_limits=y_limits,
        )

    print_metric_summary(data, metrics, x_axis=args.x_axis)


if __name__ == "__main__":
    main()
