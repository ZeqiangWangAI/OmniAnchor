# Validation evidence

Software acceptance on real weights. None of this is evidence of construct validity:
the stimuli are synthetic, and passing means the implementation scores what the policy in
[models.md](models.md) says it scores. Criterion validity against human ratings is
reported in the manuscript for specific datasets.

## What the checks establish

| Check | Meaning of a pass |
|---|---|
| unit tests | encoding, boundary, budget, calibration, analysis, IO and CLI logic on CPU fixtures |
| selected vs full logits | the position-selected scorer equals an independent full-logit forward within 0.01 nat per token |
| cache replay | warm-cache scores are bit-identical to cold scores |
| anchor reordering | adding or reordering anchors leaves existing scores unchanged |
| turn_terminated | the terminated event never exceeds the prefix probability |
| modality state | scoring an image or video does not change subsequent text scores |
| shared prefill | the optional single-token shortcut is enabled only if it matches the reference; otherwise it is disabled and the reference path is used |

## 0.2.0 demo regression (Surrey job 63446, 2026-09-18)

`scripts/hpc/demo.sbatch` on the 0.2.0 source (Qwen3.5-4B through the `qwen3_5`
adapter), node aisurrey-debug03, RTX A5000, same pinned Transformers source and
checkpoint as 0.1.0. Evidence: `docs/evidence/surrey-63446-demo/`.

| Item | Result |
|---|---|
| unit tests in the job snapshot | 294 passed |
| native scoring | 80 English + 9 Chinese items, 0 failures |
| cache replay | 80/80 hits, identical scores |
| anchor addition and reordering | 0 nat on all 4 anchors |
| selected vs full logits, modality state | passed |
| shared prefill | 0.0156 nat > 0.01, disabled (unchanged from 0.1.0) |
| peak CUDA memory | 9,184,399,872 bytes, identical to job 44314 |
| demo wall time | 259 s |

## 0.2.0 multi-model smoke (Surrey, 2026-09-18)

`scripts/hpc/models_smoke.sbatch`: for each alias, 4 text samples x 4 anchors x 2
bridges, cache replay, `turn_terminated`, one synthetic image, and the native acceptance
check with a 1024-token full-logit bound. Job 63970, node aisurrey-debug03, RTX A5000,
294 unit tests passed in the snapshot, all four models passed. Evidence:
`docs/evidence/surrey-63970-models/` (summary, per-model report and verification JSON).

| Alias | Adapter | Text items | Replay hits | Terminated minus prefix (nat) | Image | Selected vs full / modality state | Prefix tokens | Peak memory | Time |
|---|---|---|---|---|---|---|---|---|---|
| `qwen3.5-4b` | `qwen3_5` | 32 | 32 | -1.87 | scored | passed / passed | 25 | 8.50 GiB | 98 s |
| `qwen3-vl-4b` | `qwen3_vl` | 32 | 32 | -9.15 | scored | passed / passed | 20 | 8.29 GiB | 88 s |
| `qwen3-4b` | `qwen3` | 32 | 32 | -14.74 | refused explicitly | passed / not run (text only) | 24 | 7.51 GiB | 67 s |
| `smollm3-3b` | `causal_lm` | 32 | 32 | -12.89 | refused explicitly | passed / not run (text only) | 80 | 5.76 GiB | 53 s |

"Refused explicitly" means the image sample produced failed rows with an `OmniAnchorError`
naming the adapter's modalities, never a silent text-only score. The SmolLM3 prefix is
longer because its published template prepends a system preamble; with the default
thinking mode enabled that preamble exceeded the original 256-token full-logit bound
(job 63395), which is why the registry alias disables thinking and the bound is a
parameter. Raw log probabilities differ across models and are not comparable without a
shared reference.

## 0.1.0 demo evidence (Surrey, 2026-09-09)

Slurm job 44314, node aisurrey-debug03, RTX A5000 24 GB, Python 3.11.15, PyTorch
2.9.0+cu128, Transformers source `4815a0a` (5.18.0.dev0), Qwen3.5-4B revision `851bf6e`.

| Item | Result |
|---|---|
| unit tests in the job snapshot | 167 passed |
| native scoring | 80 English items (6 texts, 2 images, 1 image+text, 1 video; 4 anchors x 2 bridges) and 9 Chinese items, 0 failures |
| cache replay | 80/80 hits, identical scores |
| anchor addition and reordering | 0 nat difference on all 4 original anchors |
| selected vs full logits | 2 multi-token anchors, max error 0 nat |
| shared prefill | max error 0.0156 nat > 0.01, disabled, reference path kept |
| peak CUDA memory | 8.55 GiB |
| wall time | 177 s for the demo, 3 min 50 s for the job |

A clean CUDA environment rebuilt from `locks/cuda-reproduction-candidate-20260911-01.txt`
(job 47467) passed `pip check` and the full test suite; GPU acceptance with both Qwen
native adapters and the Qwen3-VL embedding and reranker baselines passed in job 47472.
