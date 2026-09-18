#!/bin/bash
#SBATCH --job-name=vatex-direct-reranking
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
set -euo pipefail
VL_ROOT=/mnt/fast/nobackup/scratch4weeks/zw00924/OmniAnchor-20260910
VL_RUN="$VL_ROOT/runs/vatex-reranking-${SLURM_JOB_ID}"
mkdir "$VL_RUN"
trap 'VL_EXIT=$?; printf "%s\n" "$VL_EXIT" > "$VL_RUN/exit_code.txt"' EXIT
cp -a "${VL_SOURCE_DIR:?}" "$VL_RUN/source"
cd "$VL_RUN/source"
export PYTHONPATH="$PWD/src" HF_HOME="$VL_ROOT/hf-home" TMPDIR="$VL_ROOT/tmp/$SLURM_JOB_ID"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" TOKENIZERS_PARALLELISM=false
mkdir -p "$TMPDIR"
VL_EXTRA_ARGS=()
if [ "${VL_PILOT:-0}" = 1 ]; then VL_EXTRA_ARGS+=(--pilot); fi
if [ -n "${VL_EMBEDDING_SCORES:-}" ]; then VL_EXTRA_ARGS+=(--embedding-scores "$VL_EMBEDDING_SCORES"); fi
if [ -n "${VL_FROZEN_EVALUATION:-}" ]; then VL_EXTRA_ARGS+=(--frozen-evaluation "$VL_FROZEN_EVALUATION"); fi
if [ -n "${VL_VIDEO_SHARD:-}" ]; then VL_EXTRA_ARGS+=(--video-shard "$VL_VIDEO_SHARD"); fi
"${VL_ENV_PYTHON:?}" scripts/run_vatex_reranking.py --samples "${VL_SAMPLES:?}" \
    --config configs/studies/vatex_general128.json --vision-reuse-gate "${VL_VISION_REUSE_GATE:?}" \
    --output "$VL_RUN/reranking" "${VL_EXTRA_ARGS[@]}"
