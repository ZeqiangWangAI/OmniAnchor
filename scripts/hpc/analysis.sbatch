#!/bin/bash
#SBATCH --job-name=omnianchor-statistics
#SBATCH --partition=debug,2080ti
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
set -euo pipefail
VL_ROOT=/mnt/fast/nobackup/scratch4weeks/zw00924/OmniAnchor-20260910
VL_RUN="$VL_ROOT/runs/analysis-${SLURM_JOB_ID}"
mkdir "$VL_RUN"
trap 'VL_EXIT=$?; printf "%s\n" "$VL_EXIT" > "$VL_RUN/exit_code.txt"' EXIT
cp -a "${VL_SOURCE_DIR:?}" "$VL_RUN/source"
cd "$VL_RUN/source"
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"${VL_ENV_PYTHON:?}" scripts/run_analysis_request.py --request "${VL_ANALYSIS_REQUEST:?}" --job-output "$VL_RUN"
