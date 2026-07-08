#!/usr/bin/env bash
# Inventory ASR training runs on the CIRC cluster.
#
# Emits TSV to stdout: run_id \t run_dir \t trainer_state_path \t config_path
# Intended to be piped into tools/fetch_diff.py.
#
# Usage:
#   ssh circ 'bash -s' < tools/remote_inventory.sh
#   ssh circ 'bash -s' < tools/remote_inventory.sh -- -d 14
#   ssh circ 'bash -s' < tools/remote_inventory.sh -- -d 14 -f xlsr
#
# Options: -d DAYS (default 7; 0 = no time filter), -f FILTER (substring match
# on run_id).
#
# IMPORTANT: set BASE_DIRS below to wherever your `models/` output tree lives on
# the cluster (i.e. the `output_dir` you train with). Verify this path before
# first use — it is a best guess mirroring the LAPT layout.

set -euo pipefail

BASE_DIRS=(
    "/scratch/cdowney4/ASR/models"
)
DAYS=7
FILTER=""

while getopts "d:f:" opt; do
    case "$opt" in
        d) DAYS="$OPTARG" ;;
        f) FILTER="$OPTARG" ;;
        *) echo "Usage: $0 [-d DAYS] [-f FILTER]" >&2; exit 1 ;;
    esac
done

for BASE_DIR in "${BASE_DIRS[@]}"; do
    [ -d "$BASE_DIR" ] || continue

    # A run directory is one containing a training_config.yaml. ASR run dirs are
    # models/{id}/{model}_{training} (2 levels) or models/{codename} (1 level).
    configs=$(find "$BASE_DIR" -mindepth 1 -maxdepth 3 -name training_config.yaml)

    for config in $configs; do
        run_dir=$(dirname "$config")

        # skip runs whose directory hasn't changed within the time window
        if [ "$DAYS" -ne 0 ]; then
            if [ -z "$(find "$run_dir" -maxdepth 0 -mtime "-${DAYS}" 2>/dev/null)" ]; then
                continue
            fi
        fi

        # run_id is the run dir path relative to BASE_DIR, with '/' -> '__'
        rel=${run_dir#"$BASE_DIR"/}
        run_id=${rel//\//__}

        if [ -n "$FILTER" ] && [[ "$run_id" != *"$FILTER"* ]]; then
            continue
        fi

        # find trainer_state.json: top-level if present, else latest checkpoint
        if [ -f "$run_dir/trainer_state.json" ]; then
            ts="$run_dir/trainer_state.json"
        else
            latest_ckpt=$(ls -d "$run_dir"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1) || true
            if [ -n "$latest_ckpt" ] && [ -f "$latest_ckpt/trainer_state.json" ]; then
                ts="$latest_ckpt/trainer_state.json"
            else
                continue
            fi
        fi

        printf '%s\t%s\t%s\t%s\n' "$run_id" "$rel" "$ts" "$config"
    done
done
