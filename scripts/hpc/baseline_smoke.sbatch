#!/bin/bash
#SBATCH --job-name=vlanchor-baselines32
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
set -euo pipefail
VL_CAMPAIGN=/mnt/fast/nobackup/scratch4weeks/zw00924/VLanchor-20260910
VL_RUN="$VL_CAMPAIGN/runs/baselines-${SLURM_JOB_ID}"
mkdir "$VL_RUN"
export HF_HOME="$VL_CAMPAIGN/hf-home"
export TMPDIR="$VL_CAMPAIGN/tmp/$SLURM_JOB_ID"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p "$TMPDIR"
trap 'VL_EXIT=$?; printf "%s\n" "$VL_EXIT" > "$VL_RUN/exit_code.txt"' EXIT
cp -a "$VL_CAMPAIGN/source" "$VL_RUN/source"
cd "$VL_RUN/source"
export PYTHONPATH="$PWD/src"
VL_PYTHON="${VL_ENV_PYTHON:?Supply a verified clean environment}"
"$VL_PYTHON" -m pip check
"$VL_PYTHON" -m pip freeze > "$VL_RUN/environment.txt"
for VL_LANGUAGE in en zh; do
    if [ "$VL_LANGUAGE" = en ]; then VL_DATASET=emobank-reader; else VL_DATASET=chinese-emobank-sentence; fi
    for VL_METHOD in e5 qwen-embedding qwen-reranker; do
        "$VL_PYTHON" scripts/run_baseline_shard.py --method "$VL_METHOD" \
          --config "configs/smoke/qwen35-${VL_LANGUAGE}.json" \
          --samples "data/prepared/text-smoke-20260910-02/${VL_DATASET}/smoke-samples.json" \
          --output "$VL_RUN/${VL_METHOD}-${VL_LANGUAGE}"
    done
done
