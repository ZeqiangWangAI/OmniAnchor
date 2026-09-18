#!/bin/bash
#SBATCH --job-name=vlanchor-demo
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/demo-%j.out
#SBATCH --error=logs/demo-%j.err

set -euo pipefail

VL_SOURCE_DIR="${SLURM_SUBMIT_DIR:?Submit from the VLanchor checkout}"
VL_RUN_DIR="/var/tmp/${USER}/vlanchor/${SLURM_JOB_ID}"
VL_RESULTS="${VL_SOURCE_DIR}/runs/surrey-${SLURM_JOB_ID}"
VL_BASE_PYTHON="${VL_BASE_PYTHON:-/mnt/fast/nobackup/users/zw00924/miniforge3/envs/dapo-py311/bin/python}"
mkdir -p "$VL_RUN_DIR" "$VL_RESULTS"
for VL_ENTRY in src tests scripts configs pyproject.toml; do
    if [ -e "${VL_SOURCE_DIR}/${VL_ENTRY}" ]; then
        cp -a "${VL_SOURCE_DIR}/${VL_ENTRY}" "$VL_RUN_DIR/"
    fi
done
export HF_HOME="${VL_HF_HOME:-${VL_RUN_DIR}/hf-home}"
export PIP_CACHE_DIR="${VL_RUN_DIR}/pip-cache"
export TMPDIR="${VL_RUN_DIR}/tmp"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
mkdir -p "$TMPDIR"

finish() {
    VL_EXIT_CODE=$?
    if [ -d "${VL_RUN_DIR}/results" ]; then
        cp -a "${VL_RUN_DIR}/results/." "$VL_RESULTS/"
    fi
    printf '%s\n' "$VL_EXIT_CODE" > "${VL_RESULTS}/exit_code.txt"
    exit "$VL_EXIT_CODE"
}
trap finish EXIT

hostname
date -u
df -h "$VL_RUN_DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
if [ -n "${VL_ENV_PYTHON:-}" ] && [ -x "$VL_ENV_PYTHON" ]; then
    VL_PYTHON="$VL_ENV_PYTHON"
else
    "$VL_BASE_PYTHON" -m venv --system-site-packages "${VL_RUN_DIR}/venv"
    VL_PYTHON="${VL_RUN_DIR}/venv/bin/python"
    "$VL_PYTHON" -m pip install --disable-pip-version-check --quiet --no-deps \
      'https://github.com/huggingface/transformers/archive/4815a0a6a064214f2d8208c094464a5a6b76ca8d.tar.gz'
    "$VL_PYTHON" -m pip install --disable-pip-version-check --quiet \
      av 'tokenizers>=0.23.1,<0.24.0' 'safetensors>=0.8.0'
fi
"$VL_PYTHON" -m pip freeze > "${VL_RESULTS}/environment.txt"
export PYTHONPATH="${VL_RUN_DIR}/src"
export VL_RUN_DIR VL_RESULTS
"$VL_PYTHON" - <<'PY'
import os, json, hashlib
from pathlib import Path
root=Path(os.environ['VL_RUN_DIR'])
hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ['src','scripts','tests'] for p in (root/folder).rglob('*.py')}
data={'slurm_job_id':os.environ['SLURM_JOB_ID'],'node':os.environ.get('SLURMD_NODENAME'),
      'source_sha256':hashes,'node_workdir':str(root)}
(Path(os.environ['VL_RESULTS'])/'job_manifest.json').write_text(json.dumps(data,indent=2)+'\n')
PY
"$VL_PYTHON" -m pytest "${VL_RUN_DIR}/tests" -q \
  --junitxml="${VL_RESULTS}/tests.xml" -p no:cacheprovider
"$VL_PYTHON" "${VL_RUN_DIR}/scripts/run_demo.py" --backend hf --output "${VL_RUN_DIR}/results"
