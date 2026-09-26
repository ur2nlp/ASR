#!/bin/bash
#SBATCH -c 12
#SBATCH --mem=128g
#SBATCH --gres=gpu:1
#SBATCH -t 120:00:00
#SBATCH -o outputs/%x.out
#SBATCH -e outputs/%x.err
#SBATCH --mail-type=END,FAIL

# Standard (non-preempt) training launch. Submit with a job name, which also
# names the log files (outputs/<name>.out / .err) and is a good place to reuse
# the run codename:
#
#     sbatch -J <run_name> scripts/run.sh
#
# Site-specific submission flags -- partition, account, mail address -- are
# deliberately not in this file. Keeping them out is what lets it stay
# byte-identical on every branch and in every fork, so merges never conflict
# over them. Put yours in one variable in your cluster shell rc:
#
#     export ASR_SBATCH_FLAGS="-p <partition> -A <account> --mail-user=<you@example.edu>"
#     sbatch $ASR_SBATCH_FLAGS -J <run_name> scripts/run.sh
#
# Command-line flags override #SBATCH directives, so the same variable also
# overrides the resource defaults above. With --mail-user unset, SLURM mails the
# submitting user.
#
# Choose the run with environment variables, and pass any extra Hydra overrides
# as trailing arguments:
#
#     DATASET=<your_dataset> MODEL=xls-r sbatch -J <run_name> scripts/run.sh \
#         training.learning_rate=3e-4
#
# `experiment_id` is appended as a suffix to the descriptive output path, and
# defaults to the job name.

set -euo pipefail

DATASET=${DATASET:?set DATASET to a config in configs/dataset/ (e.g. DATASET=my_corpus)}
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
