#!/bin/bash
# Run on the Surrey submit host after staging the named immutable archive.
set -euo pipefail
VL_ROOT=/mnt/fast/nobackup/scratch4weeks/zw00924/OmniAnchor-20260910
cd "$VL_ROOT/source"
tar -xzf "$VL_ROOT/omnianchor-text-final-ad08a6c.tgz"
VL_RELEASE="$VL_ROOT/releases/text-final-ad08a6c"
test ! -e "$VL_RELEASE"
cp -a "$VL_ROOT/source" "$VL_RELEASE"
cd "$VL_RELEASE"
export VL_SOURCE_DIR="$VL_RELEASE" VL_ENV_PYTHON="$VL_ROOT/runs/smoke-44672/venv/bin/python"
export PYTHONPATH="$PWD/src:$PWD/scripts"
unset VL_SHARD_DIR VL_CONFIG_DIR VL_VISION_REUSE_GATE
"$VL_ENV_PYTHON" - <<'PY'
from pathlib import Path
from frozen_evaluation import validate_frozen_evaluation
studies = [('valueeval','configs/studies/valueeval_fixed3.json'),
           ('emobank','configs/smoke/qwen35-en.json'),
           ('chinese-affect','configs/smoke/qwen35-zh.json')]
for study, config in studies:
    for method in ['native','e5','qwen-embedding','qwen-reranker']:
        validate_frozen_evaluation(Path(f'research/contracts/{study}-final-20260910-01.json'),
            Path(f'data/prepared/{study}-final-20260910-01/samples.json'), Path(config), method)
    print('Frozen preflight passed:', study, flush=True)
PY
for VL_STUDY in valueeval emobank chinese-affect; do
    case "$VL_STUDY" in
        valueeval) VL_CONFIG=configs/studies/valueeval_fixed3.json ;;
        emobank) VL_CONFIG=configs/smoke/qwen35-en.json ;;
        chinese-affect) VL_CONFIG=configs/smoke/qwen35-zh.json ;;
    esac
    export VL_CONFIG
    export VL_SAMPLES="data/prepared/$VL_STUDY-final-20260910-01/samples.json"
    export VL_FROZEN_EVALUATION="research/contracts/$VL_STUDY-final-20260910-01.json"
    sbatch --time=03:00:00 --output="$VL_ROOT/logs/$VL_STUDY-final-%j.out" --error="$VL_ROOT/logs/$VL_STUDY-final-%j.err" scripts/hpc/development.sbatch
    export VL_METHODS='e5 qwen-embedding qwen-reranker'
    sbatch --time=03:00:00 --output="$VL_ROOT/logs/$VL_STUDY-final-baselines-%j.out" --error="$VL_ROOT/logs/$VL_STUDY-final-baselines-%j.err" scripts/hpc/baseline_development.sbatch
done
