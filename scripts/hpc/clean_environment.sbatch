#!/bin/bash
#SBATCH --job-name=omnianchor-clean-env
#SBATCH --partition=debug,2080ti
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail
VL_ROOT=/mnt/fast/nobackup/scratch4weeks/zw00924/OmniAnchor-20260910
VL_SOURCE="${VL_SOURCE_DIR:?Set immutable source release}"
VL_RUN="$VL_ROOT/runs/clean-environment-${SLURM_JOB_ID}"
mkdir "$VL_RUN"
trap 'VL_EXIT=$?; printf "%s\n" "$VL_EXIT" > "$VL_RUN/exit_code.txt"' EXIT
export TMPDIR="$VL_RUN/tmp"
export PIP_CACHE_DIR="$VL_ROOT/pip-cache"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir "$TMPDIR"
df -h "$VL_ROOT" > "$VL_RUN/storage.txt"
VL_BASE=/mnt/fast/nobackup/users/zw00924/miniforge3/envs/dapo-py311/bin/python
"$VL_BASE" -m venv "$VL_RUN/venv"
VL_PYTHON="$VL_RUN/venv/bin/python"
cp "$VL_SOURCE/locks/cuda-reproduction-candidate-20260911-01.txt" "$VL_RUN/requirements.txt"
"$VL_PYTHON" -m pip install -r "$VL_RUN/requirements.txt" > "$VL_RUN/install.log" 2>&1
"$VL_PYTHON" -m pip install --no-deps "$VL_SOURCE" > "$VL_RUN/package-install.log" 2>&1
"$VL_PYTHON" -m pip check > "$VL_RUN/pip-check.txt" 2>&1
"$VL_PYTHON" -m pip freeze > "$VL_RUN/environment.txt"
cd "$VL_SOURCE"
"$VL_PYTHON" -m pytest -q --junitxml="$VL_RUN/tests.xml" -p no:cacheprovider > "$VL_RUN/tests.log" 2>&1
# This CPU-only job installs dependencies and runs toy/unit tests, never model inference.
# A separate allocated-GPU smoke must pass before this environment is accepted.
