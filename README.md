# OmniAnchor

OmniAnchor measures researcher-named concepts in text, images and video with a frozen
generative model. You name the concept (an *anchor*, e.g. `freedom`), choose a relation
sentence (a *bridge*, e.g. "A concept associated with this material is:"), and the toolkit
returns the model's log probability of the anchor as the continuation of the material plus
the bridge. Nothing is fine-tuned, no candidate list is shown to the model, and adding or
reordering anchors never changes an existing score.

An anchor coordinate is a model-conditioned continuation score, not a calibrated
probability that the material "has" the concept. Use a fixed reference set to make scores
comparable across materials, groups and modalities.

This is the toolkit behind *OmniAnchor: measuring researcher-named psychological constructs
in text, images and video* (Nature Communications, submitted 2026). Python package and
command-line entry point are both `omnianchor`.

## Install

```bash
python -m pip install -e '.[dev]'        # library, CLI, tests (CPU only)
python -m pip install -e '.[hf]'         # + torch, transformers, av for real models (CUDA)
python -m pytest -q
```

Python 3.11 or newer. Real model scoring runs on CUDA in BF16 (or explicit FP32); there is
no CPU fallback. The `toy` backend is a deterministic software fixture with no scientific
validity; use it to check a pipeline before spending GPU time.

## Quickstart

One YAML, one command, one run directory:

```bash
omnianchor run --config configs/pipelines/toy_quickstart.yaml
ls runs/toy-quickstart   # scores.parquet, reference.json, matrix.npz, analysis-*.json, run.json
```

The same chain from Python:

```python
from omnianchor import Anchor, Bridge, Part, Sample, fit_reference, score, to_matrix, transform
from omnianchor.backends import ToyBackend

samples = [Sample(id="a", parts=(Part(type="text", text="Neighbors share food."),), metadata={"split": "train"}),
           Sample(id="b", parts=(Part(type="text", text="People demand freedom."),), metadata={"split": "train"})]
anchors = [Anchor(id="care", surface="care"), Anchor(id="freedom", surface="freedom")]
bridges = [Bridge(id="b1", prefix="A concept associated with this material is:\n")]

raw = score(samples, anchors, bridges, ToyBackend())      # long table: sample x anchor x bridge
reference = fit_reference(raw)                            # fit on train/external material only
matrix = to_matrix(transform(raw, reference), variant="reference_z")
```

Swap `ToyBackend()` for a real model by loading a study file (`omnianchor.io.load_spec`)
and calling `measure(samples, spec)`.

## Configuration

A study file names the model, the anchors, the bridges and the scoring event. Anchors and
bridges can be inlined or taken from packs; the model can be a registry alias:

```yaml
name: qwen35-affect12-en
model: qwen3.5-4b                       # alias -> pinned checkpoint + adapter, or a full mapping
anchors: ../anchors/affect12_en.yaml    # pack file with a top-level `anchors:` list, or inline list
bridges: ../bridges/evokes_emotion_en.yaml
event: token_prefix                     # or turn_terminated
seed: 42
```

A pipeline file chains measure, calibrate, export and analyze:

```yaml
study: ../studies/qwen35_affect12_en.yaml
samples: my_samples.json
output_dir: runs/affect
reference: {split: train}               # or samples: external.json, or scores: ref.parquet
export: {variant: reference_z, missing: error}
analyses: [{kind: pca}, {kind: cluster, k: 3}, {kind: network}]
```

Samples are JSON or JSON Lines records with `id`, ordered `parts` (text, image path, video
path with optional clip bounds) and optional `metadata.split`, `group_id`, `time`.
See [docs/configuration.md](docs/configuration.md) for every key.

## Models

| Alias | Checkpoint | Adapter | Modalities |
|---|---|---|---|
| `qwen3.5-4b` | Qwen/Qwen3.5-4B | `qwen3_5` | text, image, video |
| `qwen3-vl-4b` | Qwen/Qwen3-VL-4B-Instruct | `qwen3_vl` | text, image, video |
| `qwen3-4b` | Qwen/Qwen3-4B | `qwen3` | text |
| `smollm3-3b` | HuggingFaceTB/SmolLM3-3B | `causal_lm` | text |
| `toy` | deterministic fixture | – | text, image, video |

Every Hugging Face alias pins an immutable revision. Any other chat model can be scored
by writing the `model:` mapping in full; text-only models use the generic `causal_lm`
adapter and must declare `turn_end_token` to score the `turn_terminated` event. Run
`omnianchor models` for the registry and `omnianchor verify-model` to check a checkpoint
against the independent full-logit reference before using it. Details, and how to add a
model family, are in [docs/models.md](docs/models.md); GPU evidence for the aliases above
is in [docs/validation.md](docs/validation.md).

## Command line

| Command | Purpose |
|---|---|
| `run` | pipeline YAML: measure, calibrate, export, analyze |
| `validate` | check a study and samples without loading weights |
| `measure` | score samples with a study; Parquet + manifest |
| `calibrate` | fit a reference; optionally transform a score table |
| `export` | average bridges into a sample x anchor matrix |
| `analyze` | cluster, pca, groups, shift, network, reliability |
| `evaluate` | multilabel, VAD, candidate and retrieval metrics against labels |
| `prepare` | convert a local public dataset with a built-in adapter |
| `optimize-bridges` | bounded subset selection from a pre-scored bridge pool |
| `models` | list aliases and adapters |
| `verify-model` | native acceptance check of a study's checkpoint on CUDA |

Every command prints one JSON object; see [docs/cli.md](docs/cli.md).

## GPU clusters

`scripts/hpc/demo.sbatch` and `scripts/hpc/models_smoke.sbatch` run the unit tests and a
real-model smoke inside a Slurm GPU allocation, keeping weights and the environment on
node-local disk. [docs/hpc.md](docs/hpc.md) explains the layout and how to adapt it.

## What a score is, and is not

Scores are the model's association between the material and the anchor under the chosen
relation. They are not human ratings, and they are not comparable across relations,
languages or models without a shared reference. Validity against human criteria is an
empirical question answered in the paper for specific datasets; the toolkit reports the
measurement, its provenance and its failures, never an accuracy.

## Paper record

`scripts/` (indexed in [scripts/README.md](scripts/README.md)), `configs/experiments`,
`configs/studies/*.json`, `configs/smoke`, `locks/` and `paper/` are the frozen record of the
manuscript's experiments and figures. They are kept as run, with their original paths; some
read private data or evidence directories that are not redistributed. Raw scores, manifests
and Source Data are released on acceptance as stated in the manuscript.

## Citation and license

Apache-2.0; see `LICENSE` and `THIRD_PARTY_NOTICES.md`. Cite the manuscript (see
`CITATION.cff`). Authors: Zeqiang Wang (University of Surrey), Hanru Qiao (Hebei
University), Yu Zhan (Huazhong Agricultural University), Yu Yue (Hong Kong Baptist
University), Zixi Chen (New York University Shanghai), Suparna De (University of Surrey),
Qingwen Xu (New York University).
