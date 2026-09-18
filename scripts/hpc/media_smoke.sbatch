#!/bin/bash
#SBATCH --job-name=vlanchor-media-smoke
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
set -euo pipefail
VL_CAMPAIGN=/mnt/fast/nobackup/scratch4weeks/zw00924/VLanchor-20260910
VL_RUN="$VL_CAMPAIGN/runs/media-smoke-${SLURM_JOB_ID}"
mkdir "$VL_RUN"
trap 'VL_EXIT=$?; printf "%s\n" "$VL_EXIT" > "$VL_RUN/exit_code.txt"' EXIT
cp -a "$VL_CAMPAIGN/source" "$VL_RUN/source"
cd "$VL_RUN/source"
export PYTHONPATH="$PWD/src"
export HF_HOME="$VL_CAMPAIGN/hf-home"
export TMPDIR="$VL_CAMPAIGN/tmp/$SLURM_JOB_ID"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p "$TMPDIR"
VL_SAMPLES=data/prepared/multimodal-smoke-20260910-02/samples.json
for VL_MODEL in qwen35 qwen3vl; do
    "${VL_ENV_PYTHON:?}" scripts/run_measurement_shard.py --config "configs/smoke/media-$VL_MODEL.json" \
      --samples "$VL_SAMPLES" --output "$VL_RUN/$VL_MODEL"
done
for VL_METHOD in qwen-embedding qwen-reranker; do
    "${VL_ENV_PYTHON:?}" scripts/run_baseline_shard.py --method "$VL_METHOD" \
      --config configs/smoke/media-qwen35.json --samples "$VL_SAMPLES" --output "$VL_RUN/$VL_METHOD"
done
