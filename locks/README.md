# Observed environments

`surrey-44314-observed.txt` is pip freeze from the actual successful CUDA demo environment.
`local-py311-observed.txt` is the local CPU test environment. These are provenance snapshots,
not claims that all inherited unrelated dependencies can be cleanly reinstalled together.
The Surrey base includes unused vLLM requiring older Transformers; VLanchor does not use it.

The verified reference pins Qwen3.5-4B revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` and Transformers source
`4815a0a6a064214f2d8208c094464a5a6b76ca8d`. Reproduce the small demo using
`scripts/hpc/demo.sbatch`. For full experiments, make a clean environment from the actual
used dependency closure, resolve it, freeze it, and rerun the same acceptance tests before
using acceleration kernels or another precision. Do not silently upgrade the checkpoint.

## Final analysis environment snapshot (2026-09-11)

`local-analysis-20260911-observed.txt` records 49 exact local package versions,
including SciencePlots 2.2.2. `local-analysis-20260911-runtime.json` records Python
and platform. A separate clean environment was successfully installed from these
pins; its tests and representative plotting verification are recorded separately.
These pins target the observed macOS analysis environment, not CUDA inference.

Rebuild the local analysis environment from the repository root:

```sh
uv venv --python 3.11 .venv-reproduction
uv pip install --python .venv-reproduction/bin/python -r locks/local-analysis-20260911-observed.txt
uv pip install --python .venv-reproduction/bin/python --no-deps .
.venv-reproduction/bin/python -m pytest -q
```

`production-environment-20260911-01.json` is a read-only inventory of the actual
81-package Surrey campaign environment (`runs/smoke-44672/venv`), distinct from
the older demo environment. It is provenance, not an independently validated
clean CUDA lock. Preserve the exact Transformers source commit and checkpoint
revisions documented above; a package version containing `dev` does not uniquely
identify its source. Do not replace GPU scoring with this local environment.

## Clean CUDA dependency rebuild acceptance

Slurm job 47467 installed the exact candidate requirements in a fresh Python 3.11
virtual environment without system-site packages. `pip check` reported no broken
requirements; all 243 repository tests passed. All 81 installed package versions
were compared with the production inventory and matched. GPU inference acceptance
passed separately: job 47472 completed both native models and official Qwen
embedding/reranker; job 47491 completed E5 on 16 English plus 16 Chinese inputs.
Evidence is in `research/clean-gpu-verification-47472.json` and
`research/clean-e5-verification-47491.json`. These are capability checks, not new
scientific validity results. The 243 tests belong to the frozen installation snapshot;
the current extracted source snapshot separately passed 250 tests.

The executed requirements are `cuda-reproduction-candidate-20260911-01.txt`.
The installed Transformers archive recorded SHA256
`b53524f805d7a9b11bd836af6f42be7bd224e316b19c25fb36ce2afbeb841219`
for source commit `4815a0a6a064214f2d8208c094464a5a6b76ca8d`.
Install this source explicitly rather than resolving its ambiguous development
version string. The repository package must be installed from the documented
source revision; the private `file://` path in pip freeze is provenance, not a
portable installation command.

## Portable CUDA installation sequence

On a Linux host with Python 3.11, install the verified dependency set from an
immutable extracted source directory. Run installation/unit tests separately from
GPU inference, as in the successful Slurm jobs. The filename retains `candidate`
for historical traceability; acceptance evidence is recorded above.

```sh
python3.11 -m venv .venv-cuda-reproduction
.venv-cuda-reproduction/bin/python -m pip install -r locks/cuda-reproduction-candidate-20260911-01.txt
.venv-cuda-reproduction/bin/python -m pip install --no-deps .
.venv-cuda-reproduction/bin/python -m pip check
.venv-cuda-reproduction/bin/python -m pytest -q
```

These package-install commands are the sequence exercised by job 47467; the
portable environment directory above replaces its job-specific scratch path.
For Surrey GPU verification, set `VL_SOURCE_DIR` to the immutable release and
`VL_ENV_PYTHON` to that environment's Python, then submit
`scripts/hpc/clean_environment_smoke.sbatch` and
`scripts/hpc/clean_e5_smoke.sbatch` through Slurm. Both scripts require the existing
prepared smoke inputs under the campaign's `source/data/prepared` directory and
use its checkpoint cache. Their cluster paths must be adapted explicitly for
another installation. They do not acquire restricted data or provide a standalone
all-study reproduction command. Do not execute model inference on a login node.
