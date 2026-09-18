#!/bin/bash
#SBATCH --job-name=vlanchor-dev
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
set -euo pipefail
VL_CAMPAIGN=/mnt/fast/nobackup/scratch4weeks/zw00924/VLanchor-20260910
VL_RUN="$VL_CAMPAIGN/runs/development-${SLURM_JOB_ID}"
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
if [ -n "${VL_SHARD_DIR:-}" ]; then
    VL_SAMPLES="$VL_SHARD_DIR/samples-$(printf '%05d' "${SLURM_ARRAY_TASK_ID:?}").json"
fi
if [ -n "${VL_CONFIG_DIR:-}" ]; then
    VL_CONFIG="$VL_CONFIG_DIR/spec-$(printf '%05d' "${SLURM_ARRAY_TASK_ID:?}").json"
fi
if [ -n "${VL_FROZEN_EVALUATION_DIR:-}" ]; then
    VL_FROZEN_EVALUATION="$VL_FROZEN_EVALUATION_DIR/contract-$(printf '%05d' "${SLURM_ARRAY_TASK_ID:?}").json"
fi
VL_EXTRA_ARGS=()
if [ -n "${VL_VISION_REUSE_GATE:-}" ]; then VL_EXTRA_ARGS+=(--vision-reuse-gate "$VL_VISION_REUSE_GATE"); fi
if [ -n "${VL_FROZEN_EVALUATION:-}" ]; then VL_EXTRA_ARGS+=(--frozen-evaluation "$VL_FROZEN_EVALUATION"); fi
"${VL_ENV_PYTHON:?}" scripts/run_measurement_shard.py \
  --config "${VL_CONFIG:?}" --samples "${VL_SAMPLES:?}" --output "$VL_RUN/measurement" "${VL_EXTRA_ARGS[@]}"
