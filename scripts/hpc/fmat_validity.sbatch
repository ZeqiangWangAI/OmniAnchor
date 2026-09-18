#!/bin/bash
#SBATCH --job-name=vlanchor-fmat-validity
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
set -euo pipefail
VL_ROOT=/mnt/fast/nobackup/scratch4weeks/zw00924/VLanchor-20260910
VL_RUN="$VL_ROOT/runs/fmat-validity-${SLURM_JOB_ID}"
mkdir "$VL_RUN"
trap 'VL_EXIT=$?; printf "%s\n" "$VL_EXIT" > "$VL_RUN/exit_code.txt"' EXIT
cp -a "${VL_SOURCE_DIR:?}" "$VL_RUN/source"
cd "$VL_RUN/source"
export PYTHONPATH="$PWD/src" HF_HOME="$VL_ROOT/hf-home" TMPDIR="$VL_ROOT/tmp/$SLURM_JOB_ID"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" TOKENIZERS_PARALLELISM=false
mkdir -p "$TMPDIR"
for VL_MODEL in qwen35 qwen3vl; do
    "${VL_ENV_PYTHON:?}" scripts/run_fmat_validity.py --data data/prepared/fmat-export-20260910-01 \
        --config "configs/smoke/$VL_MODEL-en.json" --output "$VL_RUN/$VL_MODEL"
done
for VL_STUDY in d1a d1b; do
    "${VL_ENV_PYTHON:?}" scripts/run_fmat_plc.py --config configs/smoke/qwen35-en.json \
        --data "data/prepared/fmat-export-20260910-01/$VL_STUDY.csv" --output "$VL_RUN/qwen35-plc-$VL_STUDY"
done
