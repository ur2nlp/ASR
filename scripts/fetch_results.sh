#!/bin/bash
# Fetch trainer states and configs for this project's runs into outputs/.
#
# The tools this calls are framework-general and carry no site configuration,
# so the cluster details come from the environment. Set these in your shell
# profile:
#
#   export ASR_REMOTE=<ssh host or alias>
#   export ASR_MODEL_DIRS=<remote model dir>[:<remote model dir>...]
#
# ASR_MODEL_DIRS is passed through to the remote side explicitly, because
# remote_inventory.sh runs there via `bash -s` and does not inherit this shell.
#
# Runs land in outputs/trainer_states/{experiment_id}.json and
# outputs/configs/{experiment_id}.yaml, keyed by the experiment_id in each
# run's training_config.yaml. See .claude/tools.md.

set -euo pipefail

: "${ASR_REMOTE:?set ASR_REMOTE to the ssh host holding the runs}"
: "${ASR_MODEL_DIRS:?set ASR_MODEL_DIRS to a colon-separated list of remote model dirs}"

inventory_args=()
while IFS= read -r -d ':' dir; do
    inventory_args+=(-b "$dir")
done <<< "${ASR_MODEL_DIRS}:"

ssh "$ASR_REMOTE" 'bash -s' < tools/remote_inventory.sh -- "${inventory_args[@]}" \
    | python tools/fetch_diff.py \
    | bash
