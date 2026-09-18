#!/bin/bash
#SBATCH --job-name=vlanchor-clean-smoke
#SBATCH --partition=debug
#SBATCH --gres=gpu:nvidia_rtx_5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00

set -euo pipefail
VL_ROOT=/mnt/fast/nobackup/scratch4weeks/zw00924/VLanchor-20260910
VL_SOURCE="${VL_SOURCE_DIR:?Set immutable release with official baseline source}"
VL_PYTHON="${VL_ENV_PYTHON:?Set successfully rebuilt environment}"
VL_DATA="$VL_ROOT/source/data/prepared"
VL_RUN="$VL_ROOT/runs/clean-smoke-${SLURM_JOB_ID}"
mkdir "$VL_RUN"
trap 'VL_EXIT=$?; printf "%s\n" "$VL_EXIT" > "$VL_RUN/exit_code.txt"' EXIT
export HF_HOME="$VL_ROOT/hf-home"
export TMPDIR="$VL_RUN/tmp"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TOKENIZERS_PARALLELISM=false
mkdir "$TMPDIR"
cd "$VL_SOURCE"
export PYTHONPATH="$PWD/src"
"$VL_PYTHON" -m pip check > "$VL_RUN/pip-check.txt"
"$VL_PYTHON" -m pip freeze > "$VL_RUN/environment.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$VL_RUN/hardware.txt"
for VL_MODEL in qwen35 qwen3vl; do
    for VL_LANGUAGE in en zh; do
        if [ "$VL_LANGUAGE" = en ]; then VL_DATASET=emobank-reader; else VL_DATASET=chinese-emobank-sentence; fi
        "$VL_PYTHON" scripts/run_measurement_shard.py --config "configs/smoke/$VL_MODEL-$VL_LANGUAGE.json" \
            --samples "$VL_DATA/text-smoke-20260910-02/$VL_DATASET/smoke-samples.json" --output "$VL_RUN/$VL_MODEL-$VL_LANGUAGE"
    done
    "$VL_PYTHON" scripts/run_measurement_shard.py --config "configs/smoke/media-$VL_MODEL.json" \
        --samples "$VL_DATA/multimodal-smoke-20260910-02/samples.json" --output "$VL_RUN/$VL_MODEL-media"
done
for VL_METHOD in qwen-embedding qwen-reranker; do
    "$VL_PYTHON" scripts/run_baseline_shard.py --method "$VL_METHOD" --config configs/smoke/media-qwen35.json \
        --samples "$VL_DATA/multimodal-smoke-20260910-02/samples.json" --output "$VL_RUN/$VL_METHOD-media"
done
# All inference uses the allocated GPU, frozen checkpoint configs and uncached scoring.
# Text sets and the eight-input media set retain separate coverage; no test inputs.
