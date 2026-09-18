#!/bin/bash
#SBATCH --job-name=vlanchor-clean-e5
#SBATCH --partition=debug
#SBATCH --gres=gpu:nvidia_rtx_5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
set -euo pipefail
VL_ROOT=/mnt/fast/nobackup/scratch4weeks/zw00924/VLanchor-20260910
VL_SOURCE="${VL_SOURCE_DIR:?Set immutable release}"
VL_PYTHON="${VL_ENV_PYTHON:?Set verified rebuilt environment}"
VL_RUN="$VL_ROOT/runs/clean-e5-${SLURM_JOB_ID}"
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
for VL_LANGUAGE in en zh; do
    if [ "$VL_LANGUAGE" = en ]; then VL_DATASET=emobank-reader; else VL_DATASET=chinese-emobank-sentence; fi
    "$VL_PYTHON" scripts/run_baseline_shard.py --method e5 \
        --config "configs/smoke/qwen35-$VL_LANGUAGE.json" \
        --samples "$VL_ROOT/source/data/prepared/text-smoke-20260910-02/$VL_DATASET/smoke-samples.json" \
        --output "$VL_RUN/e5-$VL_LANGUAGE"
done
