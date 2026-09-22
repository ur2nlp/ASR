"""
Plot training metrics from HuggingFace Trainer logs.

For ASR runs the metrics of interest are typically `loss`, `eval_loss`,
`eval_wer` and `eval_cer`. With external eval sets, per-set metrics are
named `eval_{name}_wer`.

Supports two input formats:
1. trainer_state.json files (recommended - proper JSON)
2. Raw log output (legacy - requires string manipulation)

Usage:
    # Single run
    python tools/training_plot.py --metric loss --state-file models/zulu/xlsr300m_l40_basic/best-checkpoint/trainer_state.json

    # Compare multiple runs (regex matched against file paths under current directory)
    python tools/training_plot.py --metric eval_wer --state-pattern "models/.*/trainer_state\\.json"

    # Use raw logs (legacy)
    python tools/training_plot.py --metric loss --log-file training.log --skip-lines 5

    # Multiple metrics
    python tools/training_plot.py --metrics loss eval_wer eval_cer --state-file path/to/trainer_state.json

    # Regex patterns for metrics (matched with re.fullmatch)
    python tools/training_plot.py --metrics "eval_.*_wer" --state-file path/to/trainer_state.json
    python tools/training_plot.py --metric "eval_.*_wer" --state-file path/to/trainer_state.json

    # Mix literal names and patterns
    python tools/training_plot.py --metrics loss "eval_.*" learning_rate --state-file path/to/trainer_state.json

    # Save to file instead of showing
    python tools/training_plot.py --metric loss --state-file path/to/trainer_state.json --output plot.png

    # Set y-axis limits shared by all subplots
    python tools/training_plot.py --metric loss --state-file path/to/trainer_state.json --ylim 0 5
    python tools/training_plot.py --metric loss --state-file path/to/trainer_state.json --ylim 0  # lower bound only

    # Set per-subplot y-axis limits (METRIC:LOWER[:UPPER]; "none" for an auto bound)
    python tools/training_plot.py --metrics loss eval_loss --state-file path/to/trainer_state.json --ylims loss:0:5 eval_loss:none:3

    # Mix a shared default (--ylim) with per-metric overrides (--ylims wins for named metrics)
    python tools/training_plot.py --metrics loss eval_loss grad_norm --state-file path/to/trainer_state.json --ylim 0 10 --ylims eval_loss:1:3
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
from plotnine import (
    aes,
    element_line,
    element_rect,
    element_text,
    facet_wrap,
    geom_blank,
    geom_line,
    geom_point,
    ggplot,
    labs,
    theme,
    theme_bw,
    theme_minimal,
    ylim,
)


def theme_colors(dark):
    """Return (background, foreground, grid, geom) colors for the plot scheme.

    plotnine has no true dark-background theme (``theme_dark()`` only darkens
    the panel), so dark mode is built by overriding the backgrounds, text,
    grid, and default line/point colors. In light mode the returned colors
    reproduce the original styling (grid stays at the base-theme default).

    Args:
        dark: If True, return a dark scheme; otherwise the light default.

    Returns:
        Tuple of ``(background, foreground, grid, geom)`` color strings. ``grid``
        and ``geom`` are None in light mode, meaning "leave the base-theme /
        geom default untouched".
    """
    if dark:
        return '#1e1e1e', '#e6e6e6', '#3a3a3a', '#e6e6e6'
    return 'white', 'black', None, None


def load_from_trainer_state(filepath):
    """Load log history from trainer_state.json (proper JSON format)."""
    with open(filepath) as f:
        state = json.load(f)
    return pd.DataFrame(state['log_history'])


def load_from_raw_log(filepath, skip_lines=0):
    """Load from raw log output (legacy - almost-JSON format)."""
    lines = open(filepath).readlines()[skip_lines:]
    # HF logs use single quotes, need to convert to double quotes for JSON
    jsonl_text = '\n'.join([line.replace('\'', '\"') for line in lines])
    return pd.read_json(jsonl_text, lines=True)


def find_files_by_regex(pattern, root='.'):
    """Walk directory tree and return files whose paths match the regex."""
    compiled = re.compile(pattern)
    matches = []
    for filepath in Path(root).rglob('*'):
        if filepath.is_file() and compiled.search(str(filepath)):
            matches.append(str(filepath))
    return sorted(matches)


def load_data(state_file=None, state_pattern=None, log_file=None, log_pattern=None, skip_lines=0, run_names=None, exclude_pattern=None):
    """Load training data from various sources."""
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
                df['run'] = run_names[idx]
            else:
                df['run'] = filepath
            dataframes.append(df)

    elif state_file:
        df = load_from_trainer_state(state_file)
        df['run'] = run_names[0] if run_names else state_file
        dataframes.append(df)

    elif log_pattern:
        files = find_files_by_regex(log_pattern)

        if exclude_re:
            files = [f for f in files if not exclude_re.search(f)]

        if not files:
            print(f"Warning: No files found matching pattern: {log_pattern}", file=sys.stderr)
        for idx, filepath in enumerate(files):
            df = load_from_raw_log(filepath, skip_lines)
            if run_names and idx < len(run_names):
                df['run'] = run_names[idx]
            else:
                df['run'] = filepath
            dataframes.append(df)

    elif log_file:
        df = load_from_raw_log(log_file, skip_lines)
        df['run'] = run_names[0] if run_names else log_file
        dataframes.append(df)

    else:
        raise ValueError("Must provide one of: --state-file, --state-pattern, --log-file, or --log-pattern")

    if not dataframes:
        raise ValueError("No data loaded. Check file paths.")

    return pd.concat(dataframes, ignore_index=True)


def plot_metric(data, metric, x_axis='step', output=None, title=None, y_limits=None, dark=False):
    """Create a plot for a single metric.

    Args:
        y_limits: Tuple of (lower, upper) for y-axis. Either can be None for auto.
        dark: If True, render on a dark background with light text and lines.
    """
    # Filter to rows where metric exists
    metric_data = data[data[metric].notna()].copy()

    if len(metric_data) == 0:
        print(f"Warning: No data found for metric '{metric}'", file=sys.stderr)
        print(f"Available metrics: {[col for col in data.columns if data[col].notna().any()]}", file=sys.stderr)
        return None

    # Determine if we're comparing multiple runs
    multiple_runs = len(metric_data['run'].unique()) > 1

    # Choose x-axis (prefer 'step' over 'epoch' if available)
    if x_axis not in metric_data.columns or metric_data[x_axis].isna().all():
        # Fallback to epoch if step not available
        x_axis = 'epoch' if 'epoch' in metric_data.columns else 'step'

    # Find min and max points to mark
    min_idx = metric_data[metric].idxmin()
    max_idx = metric_data[metric].idxmax()
    extrema_data = metric_data.loc[[min_idx, max_idx]].copy()

    background, foreground, grid, geom_color = theme_colors(dark)

    # In single-run mode the line/points have a fixed color; give them the
    # light geom color under dark mode. Multi-run mode maps color to 'run'.
    line_kwargs = {'size': 1.2}
    point_kwargs = {'size': 4, 'shape': 'x'}
    if geom_color is not None:
        line_kwargs['color'] = geom_color
        point_kwargs['color'] = geom_color

    theme_kwargs = dict(
        legend_position="bottom" if multiple_runs else "none",
        axis_title=element_text(size=14, color=foreground),
        legend_title=element_text(size=12, color=foreground),
        legend_text=element_text(size=10, color=foreground),
        axis_text=element_text(size=10, color=foreground),
        plot_title=element_text(color=foreground),
        figure_size=(10, 6),
        plot_background=element_rect(fill=background, color=background),
        panel_background=element_rect(fill=background),
    )
    if grid is not None:
        theme_kwargs['panel_grid'] = element_line(color=grid)

    plot = (
        ggplot(metric_data, aes(x=x_axis, y=metric)) +
        (geom_line(aes(color='run'), size=1.2) if multiple_runs else geom_line(**line_kwargs)) +
        (geom_point(aes(color='run'), data=extrema_data, size=4, shape='x') if multiple_runs
         else geom_point(data=extrema_data, **point_kwargs)) +
        labs(
            title=title or f'{metric} over training',
            x=x_axis.capitalize(),
            y=metric
        ) +
        theme_bw() +
        theme(**theme_kwargs)
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
    """
    Expand regex patterns to matching metric names.

    Each pattern is matched with re.fullmatch against available metrics.
    Literal metric names work as-is (they're valid regexes that match themselves).

    Args:
        patterns: List of metric names or regex patterns
        available_metrics: List of all available metric names

    Returns:
        List of matched metric names (preserving order, no duplicates)

    Examples:
        expand_metric_patterns(['eval_.*'], ['eval_loss', 'loss', 'eval_acc'])
        # Returns: ['eval_acc', 'eval_loss']

        expand_metric_patterns(['loss', 'eval_.*'], ['eval_loss', 'loss'])
        # Returns: ['loss', 'eval_loss']
    """
    expanded = []
    seen = set()

    for pattern in patterns:
        try:
            compiled = re.compile(pattern)
        except re.error as e:
            print(f"Invalid regex pattern '{pattern}': {e}", file=sys.stderr)
            sys.exit(1)
        for metric in sorted(available_metrics):
            if compiled.fullmatch(metric) and metric not in seen:
                expanded.append(metric)
                seen.add(metric)

    return expanded


def print_metric_summary(data, metrics, x_axis='step'):
    """Print summary statistics for metrics (min/max values and where they occurred)."""
    print("\n" + "="*80)
    print("METRIC SUMMARY")
    print("="*80)

    for metric in metrics:
        metric_data = data[data[metric].notna()].copy()

        if len(metric_data) == 0:
            continue

        print(f"\n{metric}:")
        print("-" * 80)

        # Find min and max
        min_idx = metric_data[metric].idxmin()
        max_idx = metric_data[metric].idxmax()

        min_val = metric_data.loc[min_idx, metric]
        max_val = metric_data.loc[max_idx, metric]
        min_step = metric_data.loc[min_idx, x_axis] if x_axis in metric_data.columns else 'N/A'
        max_step = metric_data.loc[max_idx, x_axis] if x_axis in metric_data.columns else 'N/A'
        min_run = metric_data.loc[min_idx, 'run']
        max_run = metric_data.loc[max_idx, 'run']

        print(f"  Min: {min_val:.6f} at {x_axis}={min_step} (run: {min_run})")
        print(f"  Max: {max_val:.6f} at {x_axis}={max_step} (run: {max_run})")

    print("\n" + "="*80 + "\n")


def _clip_to_window(long_data, metric, lower, upper):
    """Drop rows of a single metric whose value falls outside [lower, upper].

    Rows belonging to other metrics are left untouched. Either bound may be
    None to leave that side unclipped.
    """
    is_metric = long_data['metric_name'] == metric
    out_of_range = pd.Series(False, index=long_data.index)
    if lower is not None:
        out_of_range |= is_metric & (long_data['metric_value'] < lower)
    if upper is not None:
        out_of_range |= is_metric & (long_data['metric_value'] > upper)
    return long_data[~out_of_range]


def parse_per_metric_ylims(specs):
    """Parse ``METRIC:LOWER:UPPER`` strings into a {metric: (lower, upper)} dict.

    Each spec is ``METRIC:LOWER[:UPPER]``. A bound of ``none`` or an empty
    string means auto for that side. Examples::

        loss:0:5        -> {'loss': (0.0, 5.0)}
        eval_loss:none:3 -> {'eval_loss': (None, 3.0)}
        grad_norm:1      -> {'grad_norm': (1.0, None)}
    """
    def parse_bound(token):
        token = token.strip().lower()
        if token in ('', 'none'):
            return None
        return float(token)

    per_metric_limits = {}
    for spec in specs:
        parts = spec.split(':')
        if len(parts) == 2:
            metric, lower_token = parts
            upper_token = 'none'
        elif len(parts) == 3:
            metric, lower_token, upper_token = parts
        else:
            raise ValueError(
                f"Invalid --ylims spec '{spec}'. Expected METRIC:LOWER[:UPPER]."
            )
        per_metric_limits[metric] = (parse_bound(lower_token), parse_bound(upper_token))
    return per_metric_limits


def plot_multiple_metrics(data, metrics, x_axis='step', output=None, y_limits=None, per_metric_limits=None, dark=False):
    """Create subplots for multiple metrics.

    Args:
        y_limits: Tuple of (lower, upper) applied to *every* subplot. Either
            bound can be None for auto.
        per_metric_limits: Optional dict mapping a metric name to its own
            (lower, upper) tuple. Overrides ``y_limits`` for that subplot.
            Metrics absent from the dict fall back to ``y_limits`` (or auto).
        dark: If True, render on a dark background with light text and lines.
    """
    per_metric_limits = per_metric_limits or {}

    # Resolve the effective (lower, upper) window for each subplot: a
    # per-metric override wins, otherwise the shared y_limits, otherwise auto.
    effective_limits = {}
    for metric in metrics:
        limits = per_metric_limits.get(metric, y_limits)
        if limits:
            effective_limits[metric] = limits

    # Reshape data for faceting
    plot_data = []
    extrema_data = []

    for metric in metrics:
        metric_data = data[data[metric].notna()].copy()
        metric_data['metric_name'] = metric
        metric_data['metric_value'] = metric_data[metric]
        plot_data.append(metric_data[[x_axis, 'run', 'metric_name', 'metric_value']])

        # Find min and max points for this metric
        min_idx = metric_data[metric].idxmin()
        max_idx = metric_data[metric].idxmax()
        extrema = metric_data.loc[[min_idx, max_idx]].copy()
        extrema['metric_name'] = metric
        extrema['metric_value'] = extrema[metric]
        extrema_data.append(extrema[[x_axis, 'run', 'metric_name', 'metric_value']])

    plot_data = pd.concat(plot_data, ignore_index=True)
    extrema_data = pd.concat(extrema_data, ignore_index=True)

    # plotnine's ylim() sets a single scale shared by every facet panel, so it
    # cannot express different windows per subplot. Instead we clip the plotted
    # points to each metric's window and add invisible geom_blank anchors to
    # pin the panel range to the requested bounds. With scales='free_y' this
    # yields independent per-subplot limits.
    blank_rows = []
    for metric, (lower, upper) in effective_limits.items():
        plot_data = _clip_to_window(plot_data, metric, lower, upper)
        extrema_data = _clip_to_window(extrema_data, metric, lower, upper)
        anchor_x = plot_data[x_axis].min() if len(plot_data) else 0
        anchor_run = plot_data['run'].iloc[0] if len(plot_data) else metric
        for bound in (lower, upper):
            if bound is not None:
                blank_rows.append({
                    x_axis: anchor_x,
                    'run': anchor_run,
                    'metric_name': metric,
                    'metric_value': bound,
                })

    multiple_runs = len(plot_data['run'].unique()) > 1

    background, foreground, grid, geom_color = theme_colors(dark)

    line_kwargs = {'size': 1.0}
    point_kwargs = {'size': 3, 'shape': 'x'}
    if geom_color is not None:
        line_kwargs['color'] = geom_color
        point_kwargs['color'] = geom_color

    theme_kwargs = dict(
        legend_position="bottom" if multiple_runs else "none",
        axis_title=element_text(size=12, color=foreground),
        legend_title=element_text(size=10, color=foreground),
        legend_text=element_text(size=8, color=foreground),
        axis_text=element_text(size=8, color=foreground),
        strip_text=element_text(color=foreground),
        figure_size=(12, 4 * ((len(metrics) + 1) // 2)),
        plot_background=element_rect(fill=background, color=background),
        panel_background=element_rect(fill=background),
    )
    if grid is not None:
        theme_kwargs['panel_grid'] = element_line(color=grid)
        theme_kwargs['strip_background'] = element_rect(fill=background, color=grid)

    plot = (
        ggplot(plot_data, aes(x=x_axis, y='metric_value')) +
        (geom_line(aes(color='run'), size=1.0) if multiple_runs else geom_line(**line_kwargs)) +
        (geom_point(aes(color='run'), data=extrema_data, size=3, shape='x') if multiple_runs
         else geom_point(data=extrema_data, **point_kwargs)) +
        facet_wrap('~metric_name', scales='free_y', ncol=2) +
        labs(x=x_axis.capitalize(), y='Value') +
        theme_minimal() +
        theme(**theme_kwargs)
    )

    if blank_rows:
        plot = plot + geom_blank(data=pd.DataFrame(blank_rows))

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
        epilog=__doc__
    )

    # Input sources (mutually exclusive groups would be nice, but keeping it simple)
    parser.add_argument('--state-file', type=str, help='Path to trainer_state.json (recommended)')
    parser.add_argument('--state-pattern', type=str, help='Regex pattern for multiple trainer_state.json files (searched from cwd)')
    parser.add_argument('--log-file', type=str, help='Path to raw log file (legacy)')
    parser.add_argument('--log-pattern', type=str, help='Regex pattern for multiple raw log files (legacy, searched from cwd)')
    parser.add_argument('--skip-lines', type=int, default=0, help='Skip N lines from raw logs (default: 0)')
    parser.add_argument('--exclude-pattern', type=str, help='Regex pattern to exclude files (e.g., "v14")')

    # Metrics to plot
    parser.add_argument('--metric', type=str, help='Single metric or regex pattern (e.g., loss, eval_.*loss)')
    parser.add_argument('--metrics', nargs='+', help='Multiple metrics or regex patterns to plot as subplots')

    # Plot options
    parser.add_argument('--x-axis', type=str, default='step', choices=['step', 'epoch'],
                       help='X-axis variable (default: step)')
    parser.add_argument('--output', type=str, help='Save plot to file instead of showing')
    parser.add_argument('--title', type=str, help='Custom plot title')
    parser.add_argument('--run-names', nargs='+', help='Custom names for runs (in order of matched files)')
    parser.add_argument('--list-metrics', action='store_true', help='List available metrics and exit')
    parser.add_argument('--ylim', nargs='+', type=float, metavar='VALUE',
                       help='Y-axis limits shared by all subplots: one value for lower bound, '
                            'two for (lower, upper).')
    parser.add_argument('--ylims', nargs='+', metavar='METRIC:LOWER[:UPPER]',
                       help='Per-subplot y-axis limits, e.g. "loss:0:5 eval_loss:none:3". '
                            'Overrides --ylim for the named metrics; use "none" for an auto bound.')
    parser.add_argument('--dark', action='store_true',
                       help='Render on a dark background with light text and lines.')

    args = parser.parse_args()

    # Validate arguments
    if not any([args.state_file, args.state_pattern, args.log_file, args.log_pattern]):
        parser.error("Must provide one of: --state-file, --state-pattern, --log-file, or --log-pattern")

    if not args.list_metrics and not args.metric and not args.metrics:
        parser.error("Must provide either --metric or --metrics (or use --list-metrics)")

    # Parse y-axis limits
    y_limits = None
    if args.ylim:
        if len(args.ylim) == 1:
            y_limits = (args.ylim[0], None)  # Lower bound only
        elif len(args.ylim) == 2:
            y_limits = (args.ylim[0], args.ylim[1])
        else:
            parser.error("--ylim accepts 1 or 2 values")

    # Parse per-metric y-axis limits
    per_metric_limits = None
    if args.ylims:
        try:
            per_metric_limits = parse_per_metric_ylims(args.ylims)
        except ValueError as error:
            parser.error(str(error))

    # Load data
    data = load_data(
        state_file=args.state_file,
        state_pattern=args.state_pattern,
        log_file=args.log_file,
        log_pattern=args.log_pattern,
        skip_lines=args.skip_lines,
        run_names=args.run_names,
        exclude_pattern=args.exclude_pattern
    )

    # Get available metrics
    available_metrics = [col for col in data.columns if data[col].notna().any() and col not in ['run', 'step', 'epoch']]

    # List metrics if requested
    if args.list_metrics:
        print("Available metrics:")
        for metric in sorted(available_metrics):
            count = data[metric].notna().sum()
            print(f"  {metric} ({count} values)")
        return

    # Expand metric patterns
    patterns = args.metrics if args.metrics else [args.metric]
    metrics = expand_metric_patterns(patterns, available_metrics)

    if not metrics:
        print(f"Error: No metrics matched the pattern(s): {patterns}", file=sys.stderr)
        print(f"Available metrics: {sorted(available_metrics)}", file=sys.stderr)
        sys.exit(1)

    print(f"Plotting {len(metrics)} metric(s): {', '.join(metrics)}", file=sys.stderr)

    if len(metrics) == 1:
        # A single plot has one panel, so a per-metric override collapses to
        # that plot's y-limits; fall back to the shared --ylim otherwise.
        single_limits = y_limits
        if per_metric_limits and metrics[0] in per_metric_limits:
            single_limits = per_metric_limits[metrics[0]]
        plot_metric(data, metrics[0], x_axis=args.x_axis, output=args.output, title=args.title, y_limits=single_limits, dark=args.dark)
    else:
        plot_multiple_metrics(
            data,
            metrics,
            x_axis=args.x_axis,
            output=args.output,
            y_limits=y_limits,
            per_metric_limits=per_metric_limits,
            dark=args.dark,
        )

    print_metric_summary(data, metrics, x_axis=args.x_axis)


if __name__ == '__main__':
    main()
