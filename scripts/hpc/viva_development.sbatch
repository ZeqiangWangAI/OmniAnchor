#!/bin/bash
#SBATCH --job-name=viva-development
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
set -euo pipefail
VL_ROOT=/mnt/fast/nobackup/scratch4weeks/zw00924/OmniAnchor-20260910
VL_RUN="$VL_ROOT/runs/viva-development-${SLURM_JOB_ID}"
mkdir "$VL_RUN"
trap 'VL_EXIT=$?; printf "%s\n" "$VL_EXIT" > "$VL_RUN/exit_code.txt"' EXIT
cp -a "${VL_SOURCE_DIR:?}" "$VL_RUN/source"
cd "$VL_RUN/source"
export PYTHONPATH="$PWD/src" HF_HOME="$VL_ROOT/hf-home" TMPDIR="$VL_ROOT/tmp/$SLURM_JOB_ID"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" TOKENIZERS_PARALLELISM=false
mkdir -p "$TMPDIR"
VL_METHODS=(native qwen-embedding qwen-reranker)
VL_METHOD="${VL_METHOD:-${VL_METHODS[${SLURM_ARRAY_TASK_ID:?}]}}"
if [ -n "${VL_SHARD_DIR:-}" ]; then
    VL_SAMPLES="$VL_SHARD_DIR/samples-$(printf '%05d' "${SLURM_ARRAY_TASK_ID:?}").json"
fi
VL_EXTRA_ARGS=()
if [ -n "${VL_FROZEN_EVALUATION:-}" ]; then VL_EXTRA_ARGS+=(--frozen-evaluation "$VL_FROZEN_EVALUATION"); fi
if [ "${VL_INTERACTION_CONTROLS:-0}" = 1 ]; then VL_EXTRA_ARGS+=(--interaction-controls); fi
"${VL_ENV_PYTHON:?}" scripts/run_viva_shard.py --data data/prepared/viva-campaign-20260910-01 \
    --samples "${VL_SAMPLES:?}" --method "$VL_METHOD" --vision-reuse-gate "${VL_VISION_REUSE_GATE:?}" \
    --output "$VL_RUN/$VL_METHOD" "${VL_EXTRA_ARGS[@]}"
