#!/bin/bash
#SBATCH -p preempt
#SBATCH -c 12
#SBATCH --mem=128g
#SBATCH --gres=gpu:L40S:1
#SBATCH -t 48:00:00
#SBATCH -o outputs/%x.out
#SBATCH -e outputs/%x.err
#SBATCH --mail-user=cdowney4@ur.rochester.edu
#SBATCH --mail-type=END,FAIL

# Preempt-partition training launch on CIRC. Jobs here can be killed and
# requeued at any time, so `preempt_resume=true` is essential: on requeue the
# job auto-resumes from the latest checkpoint in the output directory instead of
# restarting from step 0. Keep the experiment id stable across requeues so the
# output directory (and its checkpoints) is reused:
#
#     sbatch -J zulu_xlsr_run1 scripts/preempt_run.sh
#
# See scripts/run.sh for the LAPT -> ASR notes (dataset has no leading `+`; conda
# env is `asr`).

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
