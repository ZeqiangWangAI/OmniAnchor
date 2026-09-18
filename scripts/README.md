# Scripts

Two scripts are part of the toolkit; everything else is the frozen record of the
manuscript's experiments, kept at the paths the Slurm launchers and tests reference. Many
of the paper scripts read acquired datasets or private evidence directories (`data/`,
`research/`) that are not redistributed, and are not meant to run from a fresh checkout.

| Toolkit | Purpose |
|---|---|
| `run_demo.py` | bounded real-model demo on synthetic stimuli (used by `hpc/demo.sbatch`) |
| `run_model_smoke.py` | multi-model smoke over the registry aliases (used by `hpc/models_smoke.sbatch`) |
| `verify_native.py` | compatibility shim for `omnianchor.verification.verify_native` |

| Paper record | Prefix |
|---|---|
| dataset acquisition and preparation | `acquire_*`, `prepare_*`, `freeze_*` |
| scoring shards and searches | `run_*` (measurement, baseline, VIVA, FMAT, bridge search, reranking) |
| analyses and statistics | `analyze_*`, `*_statistics.py`, `bootstrap_*`, `fit_*`, `evaluate_*` |
| tables and figures | `build_*`, `plot_*`, `summarize_*`, `export_*` |
| independent checks | `verify_*` (Python and R) |
| Slurm launchers and archiving | `hpc/*.sbatch`, `hpc/submit_*.py`, `hpc/archive_*.py` |
