# Changelog

## 0.2.0 (2026-09-18)

Toolkit release. The scoring policy, the study format and every existing configuration are
unchanged; everything below is additive.

- Study YAML composes reusable parts: `model: <alias>` from the packaged registry,
  `anchors:`/`bridges:` as pack paths relative to the study file (`omnianchor.config`).
- Pipeline YAML and `omnianchor run`: measure, calibrate, export and analyze into one run
  directory with a `run.json` record (`omnianchor.pipeline`).
- Model adapters (`omnianchor.backends.adapters`): `qwen3_5`, `qwen3_vl`, `qwen3` and a
  generic text-only `causal_lm` adapter selected from the checkpoint's `model_type`.
  `ModelSpec` gains optional `adapter`, `turn_end_token` and `chat_template_kwargs`; the
  backend identity records the adapter, template arguments and turn-end token.
- Registry `src/omnianchor/registry/models.yaml` with pinned aliases `qwen3.5-4b`,
  `qwen3-vl-4b`, `qwen3-4b`, `smollm3-3b`, `toy`; `omnianchor models` prints it.
- `omnianchor verify-model` runs the native acceptance check on a study's checkpoint;
  the check moved to `omnianchor.verification` (`scripts/verify_native.py` re-exports it).
- Bridge packs (`configs/bridges/*.yaml`), composed study examples, a pipeline example and
  `scripts/hpc/models_smoke.sbatch` with `scripts/run_model_smoke.py` for multi-model GPU
  verification.
- English documentation (`docs/`), continuous integration, citation metadata. Research
  status and contract documents were removed from the repository; the paper record
  (`scripts/`, `configs/experiments`, `locks/`, `paper/`) is kept at its original paths.

## 0.1.0

Package renamed from `vlanchor` to `omnianchor` (see commit `4f23360`); the runs reported
in the manuscript were produced under the development name.
