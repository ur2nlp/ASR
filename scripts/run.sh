#!/bin/bash
#SBATCH -p ur2nlp
#SBATCH -A cdowney4_lab
#SBATCH -c 12
#SBATCH --mem=128g
#SBATCH --gres=gpu:1
#SBATCH -t 120:00:00
#SBATCH -o outputs/%x.out
#SBATCH -e outputs/%x.err
#SBATCH --mail-user=cdowney4@ur.rochester.edu
#SBATCH --mail-type=END,FAIL

# Standard (non-preempt) training launch on CIRC. Submit with a job name, which
# also names the log files (outputs/<name>.out / .err) and is a good place to
# reuse the run codename:
#
#     sbatch -J zulu_xlsr_run1 scripts/run.sh
#
# Override dataset/model/training via environment variables, and pass any extra
# Hydra overrides as trailing arguments:
#
#     DATASET=zulu MODEL=xls-r sbatch -J zulu_xlsr_run1 scripts/run.sh \
#         training.learning_rate=3e-4
#
# Coming from LAPT, two things differ:
#   * `dataset` is a mandatory default in ASR, so it is `dataset=zulu` with NO
#     leading `+` (LAPT uses `+dataset=...` because dataset is not in its
#     defaults).
#   * The conda env is `asr`, not `lapt`.
# `experiment_id` works exactly as in LAPT: it is appended as a suffix to the
# descriptive output path. It defaults to the job name below.

set -euo pipefail

DATASET=${DATASET:-zulu}
MODEL=${MODEL:-xls-r}
TRAINING=${TRAINING:-ctc-basic}
EXPERIMENT_ID=${EXPERIMENT_ID:-${SLURM_JOB_NAME:-}}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
eval "$(conda shell.bash hook)"
echo "conda initialized"
conda activate asr
echo "environment activated"

experiment_id_override=()
if [ -n "$EXPERIMENT_ID" ]; then
    experiment_id_override=(experiment_id="$EXPERIMENT_ID")
fi

python -u -m src \
    dataset="$DATASET" \
    model="$MODEL" \
    training="$TRAINING" \
    preempt_resume=true \
    "${experiment_id_override[@]}" \
    "$@"
