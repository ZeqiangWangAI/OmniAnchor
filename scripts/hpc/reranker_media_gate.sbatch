#!/bin/bash
#SBATCH --job-name=reranker-media-reuse-gate
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
set -euo pipefail
VL_ROOT=/mnt/fast/nobackup/scratch4weeks/zw00924/OmniAnchor-20260910
VL_RUN="$VL_ROOT/runs/reranker-media-gate-${SLURM_JOB_ID}"
mkdir "$VL_RUN"
trap 'VL_EXIT=$?; printf "%s\n" "$VL_EXIT" > "$VL_RUN/exit_code.txt"' EXIT
cp -a "${VL_SOURCE_DIR:?}" "$VL_RUN/source"
cd "$VL_RUN/source"
export PYTHONPATH="$PWD/src" HF_HOME="$VL_ROOT/hf-home" TMPDIR="$VL_ROOT/tmp/$SLURM_JOB_ID"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" TOKENIZERS_PARALLELISM=false
mkdir -p "$TMPDIR"
"${VL_ENV_PYTHON:?}" scripts/verify_reranker_media_reuse.py --prior-gate "$VL_ROOT/runs/vision-reuse-gate-44786/gate" --output "$VL_RUN/gate"
