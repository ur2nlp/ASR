#!/bin/bash
# Upgrade lapt-core in the active environment, and nothing else.
#
#     bash scripts/update_lapt_core.sh            # the tag environment.yml pins
#     bash scripts/update_lapt_core.sh 0.1.6      # a specific release
#
# The requirement string is read out of environment.yml rather than repeated
# here, so the extras and the URL cannot drift from what the environment
# declares. With no argument this reinstalls whatever tag is currently pinned,
# which is what you want right after `git pull` moves it.
#
# `--no-deps` is deliberate. lapt-core's extras (datasets, transformers,
# sentencepiece, pandas, plotnine) are already installed from environment.yml,
# and letting pip re-resolve them can put a CPU build of torch over a working
# CUDA one.

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
env_file="$repo_root/environment.yml"

requirement=$(grep -m1 -E '^[[:space:]]*-[[:space:]]*lapt-core\[' "$env_file" \
    | sed 's/^[[:space:]]*-[[:space:]]*//')
if [ -z "$requirement" ]; then
    echo "error: no lapt-core requirement found in $env_file" >&2
    exit 1
fi

# An explicit tag may be given as 0.1.6, v0.1.6 or lapt-core-v0.1.6
if [ $# -gt 0 ]; then
    tag=$1
    case "$tag" in
        lapt-core-v*) ;;
        v*)           tag="lapt-core-$tag" ;;
        *)            tag="lapt-core-v$tag" ;;
    esac
    requirement=${requirement//lapt-core-v[0-9]*.tar.gz/$tag.tar.gz}
fi

if [ "${CONDA_DEFAULT_ENV:-}" != "asr" ]; then
    echo "warning: CONDA_DEFAULT_ENV is '${CONDA_DEFAULT_ENV:-unset}', not 'asr'" >&2
fi

echo "installing: $requirement"
python -m pip install --no-deps --upgrade "$requirement"
python -c "import importlib.metadata as m; print('lapt-core now', m.version('lapt-core'))"
