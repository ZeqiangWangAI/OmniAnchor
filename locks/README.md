# Environment locks

Provenance snapshots of the environments that produced the manuscript's runs, kept at
their original names. They are `pip freeze` outputs and inventories, not resolver-verified
universal locks; the Surrey base environment they inherit from includes unused packages.

| File | Environment |
|---|---|
| `surrey-44314-observed.txt` | CUDA demo environment of Slurm job 44314 (Qwen3.5-4B acceptance) |
| `production-environment-20260911-01.json` | 81-package inventory of the Surrey campaign environment |
| `cuda-reproduction-candidate-20260911-01.txt` | clean CUDA rebuild verified by job 47467 (`pip check` clean, all tests passed) |
| `local-py311-observed.txt`, `local-analysis-20260911-observed.txt`, `local-analysis-20260911-runtime.json` | local CPU test and analysis environments (macOS) |

Pinned sources: Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`;
Transformers source commit `4815a0a6a064214f2d8208c094464a5a6b76ca8d` (archive SHA256
`b53524f805d7a9b11bd836af6f42be7bd224e316b19c25fb36ce2afbeb841219`), which reports the
ambiguous version `5.18.0.dev0` and must be installed by commit.

Portable CUDA rebuild on Linux with Python 3.11:

```sh
python3.11 -m venv .venv-cuda
.venv-cuda/bin/python -m pip install -r locks/cuda-reproduction-candidate-20260911-01.txt
.venv-cuda/bin/python -m pip install --no-deps .
.venv-cuda/bin/python -m pip check && .venv-cuda/bin/python -m pytest -q
```

Rerun `omnianchor verify-model` on the target GPU before using acceleration kernels or a
different precision.
