#!/bin/bash
#SBATCH --job-name=vlanchor-fla-gate
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
set -euo pipefail
VL_CAMPAIGN=/mnt/fast/nobackup/scratch4weeks/zw00924/VLanchor-20260910
VL_RUN="$VL_CAMPAIGN/runs/fla-gate-${SLURM_JOB_ID}"
mkdir "$VL_RUN"
trap 'VL_EXIT=$?; printf "%s\n" "$VL_EXIT" > "$VL_RUN/exit_code.txt"' EXIT
cp -a "${VL_SOURCE_DIR:?}" "$VL_RUN/source"
cd "$VL_RUN/source"
export HF_HOME="$VL_CAMPAIGN/hf-home"
export TMPDIR="$VL_CAMPAIGN/tmp/$SLURM_JOB_ID"
export TRITON_CACHE_DIR="$VL_RUN/triton-cache"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PIP_CACHE_DIR="$VL_CAMPAIGN/pip-cache"
mkdir -p "$TMPDIR"
"${VL_ENV_PYTHON:?}" -m pip install --no-deps --target "$VL_RUN/overlay" \
  flash-linear-attention==0.5.2 fla-core==0.5.2 einops==0.8.1 > "$VL_RUN/install.log" 2>&1
export PYTHONPATH="$VL_RUN/overlay:$PWD/src"
"${VL_ENV_PYTHON:?}" -m pip check > "$VL_RUN/pip-check.txt"
"${VL_ENV_PYTHON:?}" -m pip freeze > "$VL_RUN/environment.txt"
"${VL_ENV_PYTHON:?}" scripts/verify_fla_equivalence.py --output "$VL_RUN/gate"
