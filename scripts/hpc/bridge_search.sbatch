#!/bin/bash
#SBATCH --job-name=vlanchor-train-search
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
set -euo pipefail
VL_CAMPAIGN=/mnt/fast/nobackup/scratch4weeks/zw00924/VLanchor-20260910
VL_RUN="$VL_CAMPAIGN/runs/bridge-search-${SLURM_JOB_ID}"
mkdir "$VL_RUN"
trap 'VL_EXIT=$?; printf "%s\n" "$VL_EXIT" > "$VL_RUN/exit_code.txt"' EXIT
cp -a "${VL_SOURCE_DIR:-$VL_CAMPAIGN/source}" "$VL_RUN/source"
cd "$VL_RUN/source"
export PYTHONPATH="$PWD/src"
export HF_HOME="$VL_CAMPAIGN/hf-home"
export TMPDIR="$VL_CAMPAIGN/tmp/$SLURM_JOB_ID"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p "$TMPDIR"
VL_EXTRA_ARGS=()
if [ -n "${VL_SEED_CACHE:-}" ]; then VL_EXTRA_ARGS+=(--seed-cache "$VL_SEED_CACHE"); fi
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    VL_OBJECTIVES=(validity reliability alpha random)
    VL_OBJECTIVE="${VL_OBJECTIVES[$SLURM_ARRAY_TASK_ID]}"
fi
"${VL_ENV_PYTHON:?}" scripts/run_bridge_search.py \
  --data data/prepared/valueeval-20260910-01 --output "$VL_RUN/search" \
  --objective "${VL_OBJECTIVE:-reference_guided}" "${VL_EXTRA_ARGS[@]}"
