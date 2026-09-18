#!/bin/bash
#SBATCH --job-name=vlanchor-smoke32
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00

set -euo pipefail
VL_CAMPAIGN=/mnt/fast/nobackup/scratch4weeks/zw00924/VLanchor-20260910
VL_RUN="$VL_CAMPAIGN/runs/smoke-${SLURM_JOB_ID}"
mkdir "$VL_RUN"
export TMPDIR="$VL_CAMPAIGN/tmp/$SLURM_JOB_ID"
export HF_HOME="$VL_CAMPAIGN/hf-home"
export PIP_CACHE_DIR="$VL_CAMPAIGN/pip-cache"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$TMPDIR"
trap 'VL_EXIT=$?; printf "%s\n" "$VL_EXIT" > "$VL_RUN/exit_code.txt"' EXIT
cp -a "$VL_CAMPAIGN/source" "$VL_RUN/source"
cd "$VL_RUN/source"
for VL_DATASET in emobank-reader chinese-emobank-sentence; do
    test -s "data/prepared/text-smoke-20260910-02/$VL_DATASET/smoke-samples.json"
done
df -h "$VL_CAMPAIGN"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
VL_BASE=/mnt/fast/nobackup/users/zw00924/miniforge3/envs/dapo-py311/bin/python
VL_PYTHON="${VL_ENV_PYTHON:-$VL_RUN/venv/bin/python}"
if [ -z "${VL_ENV_PYTHON:-}" ]; then
"$VL_BASE" -m venv "$VL_RUN/venv"
"$VL_PYTHON" -m pip install --disable-pip-version-check \
  torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
"$VL_PYTHON" -m pip install --disable-pip-version-check \
  'https://github.com/huggingface/transformers/archive/4815a0a6a064214f2d8208c094464a5a6b76ca8d.tar.gz' \
  'tokenizers==0.23.2' 'safetensors>=0.8.0' av accelerate qwen-vl-utils '.[dev]'
fi
"$VL_PYTHON" -m pip check
"$VL_PYTHON" -m pip freeze > "$VL_RUN/environment.txt"
export PYTHONPATH="$PWD/src"
"$VL_PYTHON" -m pytest -q --junitxml="$VL_RUN/tests.xml" -p no:cacheprovider
for VL_MODEL in qwen35 qwen3vl; do
    for VL_LANGUAGE in en zh; do
        if [ "$VL_LANGUAGE" = en ]; then VL_DATASET=emobank-reader; else VL_DATASET=chinese-emobank-sentence; fi
        "$VL_PYTHON" scripts/run_measurement_shard.py \
          --config "configs/smoke/${VL_MODEL}-${VL_LANGUAGE}.json" \
          --samples "data/prepared/text-smoke-20260910-02/${VL_DATASET}/smoke-samples.json" \
          --output "$VL_RUN/${VL_MODEL}-${VL_LANGUAGE}"
    done
done
