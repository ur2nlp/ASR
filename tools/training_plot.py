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
    python tools/training_plot.py --metric loss --state-file models/zulu/xlsr300m_ctc-basic/best-checkpoint/trainer_state.json

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

    # Average seed replicates into one curve per group (regex matched against run labels),
    # shading the members' range; runs matched by no group are drawn individually
    python tools/training_plot.py --metric eval_wer --state-pattern "outputs/trainer_states/.*\\.json" \\
        --group "ctc=ctc(-s\\d)?\\.json" --group "whisper=whisper(-s\\d)?\\.json" --band minmax

    # Read options from a YAML file (keys are option names); command-line options override it
    python tools/training_plot.py --config plot.yaml --output plot.png

Example plot.yaml:

    state_pattern: 'outputs/trainer_states/(ctc|whisper)(-s\\d)?\\.json'
    metrics: [loss, eval_wer, 'eval_.*_wer']
    groups:
      ctc: 'ctc(-s\\d)?\\.json'
      whisper: ['whisper\\.json', 'whisper-s[23]\\.json']
    band: minmax           # none | std | minmax
    ylims:
      eval_wer: [0, 0.8]
      loss: [null, 3]      # null = auto bound
"""

from lapt_core.plotting import main


if __name__ == '__main__':
    main(epilog=__doc__)
