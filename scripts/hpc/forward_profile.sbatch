#!/bin/bash
#SBATCH --job-name=vlanchor-forward-profile
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
set -euo pipefail
VL_CAMPAIGN=/mnt/fast/nobackup/scratch4weeks/zw00924/VLanchor-20260910
VL_RUN="$VL_CAMPAIGN/runs/forward-profile-${SLURM_JOB_ID}"
mkdir "$VL_RUN"
trap 'VL_EXIT=$?; printf "%s\n" "$VL_EXIT" > "$VL_RUN/exit_code.txt"' EXIT
cp -a "${VL_SOURCE_DIR:?}" "$VL_RUN/source"
cd "$VL_RUN/source"
export PYTHONPATH="$PWD/src"
export HF_HOME="$VL_CAMPAIGN/hf-home"
export TMPDIR="$VL_CAMPAIGN/tmp/$SLURM_JOB_ID"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p "$TMPDIR"
"${VL_ENV_PYTHON:?}" scripts/profile_native_forward.py --output "$VL_RUN/profile"
