# Running on a Slurm GPU cluster

Two launchers in `scripts/hpc/` run the test suite and a real-model smoke inside one GPU
allocation. Both were developed on the University of Surrey AI cluster (RTX A5000 24 GB,
debug partition) and copy only small evidence back to the shared file system.

| Launcher | What it does |
|---|---|
| `demo.sbatch` | unit tests, then `scripts/run_demo.py`: Qwen3.5-4B on synthetic text, images, a video and Chinese text, calibration, analyses, cache replay and the native acceptance check |
| `models_smoke.sbatch` | unit tests, then `scripts/run_model_smoke.py` over every alias in `configs/smoke/models.yaml`: text scoring, cache replay, `turn_terminated`, one image, native acceptance |

## Layout

The job copies `src`, `tests`, `scripts`, `configs`, `examples` and `pyproject.toml` into
`/var/tmp/$USER/omnianchor/$SLURM_JOB_ID` and records SHA256 hashes of that snapshot in
`job_manifest.json`, so later edits to the checkout cannot change a running job. The
Python environment, pip cache and Hugging Face weights also live there. On exit, results
are copied to `runs/surrey-<jobid>/` next to the checkout with `exit_code.txt`.

Environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `VL_BASE_PYTHON` | a Python 3.11 with torch + CUDA | interpreter used to create the venv (`--system-site-packages`) |
| `VL_TRANSFORMERS_COMMIT` | `4815a0a6a064214f2d8208c094464a5a6b76ca8d` | Transformers source commit installed into the venv |
| `VL_ENV_PYTHON` | unset | reuse an existing venv instead of building one |
| `VL_HF_HOME` | node-local | reuse an existing weight cache |

## Submit, watch, fetch

```bash
rsync -az --exclude=.git --exclude=runs --exclude=__pycache__ ./ cluster:/scratch/you/OmniAnchor/
ssh cluster 'cd /scratch/you/OmniAnchor && mkdir -p logs && sbatch --parsable scripts/hpc/models_smoke.sbatch'
ssh cluster 'squeue -j JOBID; tail -20 /scratch/you/OmniAnchor/logs/models-JOBID.out'
ssh cluster 'sacct -j JOBID --format=JobID,State,Elapsed,ExitCode'
rsync -az cluster:/scratch/you/OmniAnchor/runs/surrey-JOBID/ runs/surrey-JOBID/
```

Judge a job by `exit_code.txt`, `tests.xml` and the report JSON (`demo_report.json` or
`models_summary.json`), not by its disappearance from the queue.

## Adapting to another cluster

Change the `#SBATCH` header (partition, GPU type, time), `VL_BASE_PYTHON`, and the
node-local path if `/var/tmp` is not local. Keep weights out of quota-limited home
directories. For production runs build a clean environment from `locks/` and rerun the
acceptance check before enabling any acceleration kernel or another precision.

## Etiquette

The launchers only read `squeue`, `sinfo` and `nvidia-smi`; they never stop other jobs.
On out-of-memory the backend stops issuing model calls, keeps the successful rows and marks
the rest failed. It does not reduce resolution, frames or precision on its own; a smaller
budget is a new study with a new measurement identity.
