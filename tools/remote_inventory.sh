#!/usr/bin/env bash
# Inventory ASR training runs on a remote cluster.
#
# Outputs TSV to stdout: experiment_id \t dir_name \t trainer_state_path \t config_path
# Intended to be piped into tools/fetch_diff.py.
#
# This script executes on the *remote* host, so it cannot read the local
# environment. The directories to scan are therefore arguments, passed either as
# repeated -b flags or, if the remote shell exports it, as a colon-separated
# ASR_MODEL_DIRS. There is no built-in default: a path that is right for one
# cluster account is wrong for every other one.
#
# Usage:
#   ssh "$ASR_REMOTE" 'bash -s' < tools/remote_inventory.sh -- -b /path/to/models
#   ssh "$ASR_REMOTE" 'bash -s' < tools/remote_inventory.sh -- -b /a -b /b -d 14
#   ssh "$ASR_REMOTE" 'bash -s' < tools/remote_inventory.sh -- -d 14 -f whisper
#
# Options: -b DIR (repeatable), -d DAYS (default 7; 0 = no time filter),
# -f FILTER (substring match on experiment_id).

set -euo pipefail

BASE_DIRS=()
DAYS=7
FILTER=""

while getopts "b:d:f:" opt; do
    case "$opt" in
        b) BASE_DIRS+=("$OPTARG") ;;
        d) DAYS="$OPTARG" ;;
        f) FILTER="$OPTARG" ;;
        *) echo "Usage: $0 [-b DIR]... [-d DAYS] [-f FILTER]" >&2; exit 1 ;;
    esac
done

# fall back to the remote environment when no -b was given
if [ ${#BASE_DIRS[@]} -eq 0 ] && [ -n "${ASR_MODEL_DIRS:-}" ]; then
    IFS=':' read -r -a BASE_DIRS <<< "$ASR_MODEL_DIRS"
fi

if [ ${#BASE_DIRS[@]} -eq 0 ]; then
    echo "No model directories given. Pass -b DIR (repeatable), or export" >&2
    echo "ASR_MODEL_DIRS as a colon-separated list on the remote host." >&2
    exit 1
fi

for BASE_DIR in "${BASE_DIRS[@]}"; do
    [ -d "$BASE_DIR" ] || continue

    # A run directory is one containing a training_config.yaml. Unlike LAPT's
    # flat models/{run}/ tree, ASR run dirs sit at variable depth --
    # models/{dataset_id}/{model}_{training}_{experiment_id} (2 levels) or
    # models/{codename} (1 level) -- so they are found by locating the config
    # rather than by globbing one level down.
    configs=$(find "$BASE_DIR" -mindepth 1 -maxdepth 3 -name training_config.yaml)

    for config in $configs; do
        run_dir=$(dirname "$config")
        dir_name=${run_dir#"$BASE_DIR"/}

        # skip runs whose directory hasn't changed within the time window
        if [ "$DAYS" -ne 0 ]; then
            if [ -z "$(find "$run_dir" -maxdepth 0 -mtime "-${DAYS}" 2>/dev/null)" ]; then
                continue
            fi
        fi

        # The experiment id comes from the config, not from the path. Deriving
        # it from the directory instead makes the id depend on which -b the
        # caller happened to pass: the same run inventoried from models/ and
        # from models/zulu/ produces two different ids, and the fetcher cannot
        # tell they are the same run. That is how a local mirror ends up with
        # `whisper5` and `zulu__..._focus-v4k_whisper5` side by side.
        exp_id=$(grep "^experiment_id:" "$config" | awk '{print $2}') || true
        if [ -z "$exp_id" ]; then
            continue
        fi

        if [ -n "$FILTER" ] && [[ "$exp_id" != *"$FILTER"* ]]; then
            continue
        fi

        # find trainer_state.json: top-level if finished, latest checkpoint otherwise
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

        printf '%s\t%s\t%s\t%s\n' "$exp_id" "$dir_name" "$ts" "$config"
    done
done
