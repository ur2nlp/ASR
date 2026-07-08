#!/usr/bin/env bash
# Fetch ASR trainer states + configs from the CIRC cluster into outputs/fetched/.
#
# Runs the remote inventory over SSH, diffs it against the local mirror, and
# executes the resulting scp commands. See .claude/tools.md for details and
# .claude/workflows.md for the surrounding workflow.
#
# Prerequisite: an SSH host alias `circ` in ~/.ssh/config, and a correct
# BASE_DIRS path inside tools/remote_inventory.sh.
set -euo pipefail

ssh circ 'bash -s' < tools/remote_inventory.sh | python tools/fetch_diff.py | bash
