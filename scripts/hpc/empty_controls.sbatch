#!/bin/bash
#SBATCH --job-name=omnianchor-empty-controls
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
set -euo pipefail
VL_CAMPAIGN=/mnt/fast/nobackup/scratch4weeks/zw00924/OmniAnchor-20260910
VL_RUN="$VL_CAMPAIGN/runs/empty-controls-${SLURM_JOB_ID}"
mkdir "$VL_RUN"
trap 'VL_EXIT=$?; printf "%s\n" "$VL_EXIT" > "$VL_RUN/exit_code.txt"' EXIT
cp -a "${VL_SOURCE_DIR:?}" "$VL_RUN/source"
cd "$VL_RUN/source"
export PYTHONPATH="$PWD/src"
export HF_HOME="$VL_CAMPAIGN/hf-home"
export TMPDIR="$VL_CAMPAIGN/tmp/$SLURM_JOB_ID"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p "$TMPDIR"
for VL_STUDY in valueeval en zh; do
    if [ "$VL_STUDY" = valueeval ]; then VL_CONFIG=configs/studies/valueeval_fixed3.json; else VL_CONFIG="configs/smoke/qwen35-$VL_STUDY.json"; fi
    "${VL_ENV_PYTHON:?}" scripts/run_measurement_shard.py --config "$VL_CONFIG" \
      --samples data/prepared/empty-control-20260910/samples.json --output "$VL_RUN/$VL_STUDY/native"
    for VL_METHOD in e5 qwen-embedding qwen-reranker; do
        "${VL_ENV_PYTHON:?}" scripts/run_baseline_shard.py --method "$VL_METHOD" --config "$VL_CONFIG" \
          --samples data/prepared/empty-control-20260910/samples.json --output "$VL_RUN/$VL_STUDY/$VL_METHOD"
    done
done
