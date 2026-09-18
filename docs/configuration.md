# Configuration reference

All files are YAML (JSON is accepted). Unknown keys are rejected. Relative paths inside a
file resolve against that file's directory.

## Study file

| Key | Type | Default | Meaning |
|---|---|---|---|
| `name` | string | `study` | label recorded in every manifest |
| `model` | alias or mapping | `qwen3.5-4b` equivalent | see below |
| `anchors` | list or pack path | required | concept surfaces to score |
| `bridges` | list or pack path | required | relation wordings; one relation and language per study |
| `event` | `token_prefix` / `turn_terminated` | `token_prefix` | score the anchor tokens only, or anchor plus the model's turn-end token |
| `resources` | mapping | see below | pixel, frame and token budgets |
| `seed` | int | 42 | toy backend and analysis seed |
| `include_token_details` | bool | false | keep per-token ids and log probabilities in the score table |
| `system_prompt` | string | none | optional system turn before the material |

### `model`

Either a registry alias (`omnianchor models` lists them) or a mapping:

| Key | Default | Meaning |
|---|---|---|
| `backend` | `hf` | `hf` (Hugging Face, CUDA) or `toy` |
| `id` | `Qwen/Qwen3.5-4B` | Hugging Face repository |
| `revision` | pinned | 40-character commit; tags and branches are refused |
| `device` | `cuda` | `cuda`, `cuda:N`; `cpu` is refused for real models |
| `precision` | `bf16` | `bf16` or `fp32` |
| `attention_implementation` | none | passed to Transformers when set |
| `adapter` | auto | `qwen3_5`, `qwen3_vl`, `qwen3`, `causal_lm`; auto-selected from the checkpoint's `model_type` |
| `turn_end_token` | adapter default | exact token appended by `turn_terminated`; required for `causal_lm` |
| `chat_template_kwargs` | adapter default | extra chat-template arguments, e.g. `{enable_thinking: false}` |

### Anchor packs

A pack is a YAML file with a top-level `anchors:` list. Each anchor has `id`, `surface`
(exact string, never normalised or trimmed), optional `language` and `concept_id`. Packs
in `configs/anchors` include `affect12_en`, `affect12_zh` and `values20_en`.

### Bridge packs

A YAML file with a top-level `bridges:` list. Each bridge has `id`, `prefix` (exact text,
usually ending with a newline), `relation` and `language`; all bridges in a study must share
one relation and language. Packs in `configs/bridges` include `associated_en`,
`associated_zh` and `evokes_emotion_en`.

### `resources`

```yaml
resources:
  image: {min_pixels: 65536, max_pixels: 262144}
  video: {sampled_frames: 8, sampling: uniform, max_pixels_per_frame: 65536}
  limits: {input_text_tokens: 2048, anchor_continuation_tokens: 32,
           total_postprocessor_tokens: 8192, implicit_truncation: false}
```

Budgets are enforced, never applied silently: an input over budget fails with
`BudgetExceeded` and is recorded as a failed row. Changing a budget changes the
measurement identity.

## Samples

A JSON array or JSON Lines file of records:

```json
{"id": "s1",
 "parts": [{"type": "image", "path": "images/s1.jpg"},
           {"type": "text", "text": "Caption shown with the image."}],
 "language": "en", "group_id": "participant-07", "time": "2024",
 "metadata": {"split": "train"}}
```

`parts` are presented to the model in order. Media paths resolve relative to the samples
file. Video parts accept `clip_start` and `clip_end` in seconds. `metadata.split` is read by
calibration: references must be `train` or `external`, never test or validation material.
`group_id` and `time` feed the group comparison and semantic-shift analyses.

## Pipeline file

| Key | Type | Meaning |
|---|---|---|
| `study` | path or inline study mapping | what to measure with |
| `samples` | path | samples file |
| `output_dir` | path | created if missing; artifacts and `run.json` land here |
| `cache` | path | optional per-candidate score cache shared across runs |
| `reference` | mapping | exactly one of `split: train`, `split: external`, `samples: path`, `scores: path` |
| `export.variant` | `raw_logp`, `mean_token_logp`, `reference_log_ratio`, `reference_z` | matrix variant; reference variants need `reference` |
| `export.missing` | `error`, `drop_samples`, `drop_anchors` | policy for undefined coordinates |
| `analyses` | list | entries with `kind` and, where relevant, `k`, `network_kind`, `bootstrap`, `seed` |

`reference.split: train` fits on the study samples whose `metadata.split` is `train`.
`reference.samples` measures a separate external file with the same study.
`reference.scores` reuses a saved score table. `run.json` records the resolved study, the
reference provenance, matrix coverage and the runtime.

## Artifacts

| File | Content |
|---|---|
| `scores.parquet` + `.manifest.json` | one row per sample x anchor x bridge, including failed rows with their status and error |
| `reference.json` | per anchor x bridge reference statistics and their provenance |
| `calibrated.parquet` | scores with `reference_log_ratio` and `reference_z` appended |
| `matrix.npz` + `.manifest.json` | bridges averaged; `values`, `sample_ids`, `anchor_ids` |
| `analysis-<kind>.json` | serialisable analysis result |

Manifests carry the model identity (checkpoint, revision, adapter, library versions), the
resource budgets, per-sample content hashes and the compiled prompt hashes, so a score can
be traced to exactly what the model saw.
