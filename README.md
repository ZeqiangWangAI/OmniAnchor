# OmniAnchor

OmniAnchor is the measurement toolkit described in the manuscript *OmniAnchor: measuring researcher-named psychological constructs in text, images and
video with a frozen generative model* (Nature Communications, submitted 2026). The Python package and command-line
entry point keep their development name `omnianchor`; the import path and the frozen study
configurations below are unchanged from the runs reported in the paper.

This repository holds the source, tests, frozen study configurations, analysis scripts,
environment locks and documentation. Raw scores, manifests and Source Data are released
separately on acceptance, as stated in the manuscript's Data availability section.

---

# OmniAnchor

Probabilistic anchor representations for text, images, video and combined input.
An anchor coordinate is a model-conditioned continuation score, not a calibrated
probability that a document has a value. Anchors are arbitrary and need not be paired.

## Install and test

```bash
python -m pip install -e '.[dev]'
python -m pytest
python -m omnianchor --help
```

Real Qwen inference additionally needs `.[hf]` on CUDA. The research reference pins
Transformers source to `4815a0a6a064214f2d8208c094464a5a6b76ca8d`; the Surrey script
installs it into an isolated environment. No production CPU fallback is performed.
The deterministic toy backend is only a software fixture and has no scientific validity.

## Minimal Python workflow

```python
from omnianchor import Anchor, Bridge, Sample, Part, score, fit_reference, transform, to_matrix
from omnianchor.backends import ToyBackend

samples = [
    Sample(id='a', parts=(Part(type='text', text='Neighbors share food.'),), metadata={'split': 'train'}),
    Sample(id='b', parts=(Part(type='text', text='People demand freedom.'),), metadata={'split': 'train'}),
    Sample(id='c', parts=(Part(type='text', text='A locked gate protects the building.'),), metadata={'split': 'train'}),
]
anchors = [Anchor(id='care', surface='care'), Anchor(id='freedom', surface='freedom')]
bridges = [Bridge(id='b1', prefix='A concept associated with this material is:\n')]
raw = score(samples, anchors, bridges, ToyBackend())
reference = fit_reference(raw)
matrix = to_matrix(transform(raw, reference), variant='reference_z')
```

Use a separate frozen reference set for real evaluation. Do not fit it on test
data or separately for each compared group/time period. Preserve the full score
table and its `.manifest.json` sidecar with every exported matrix.

## Surrey demo

`scripts/hpc/demo.sbatch` runs unit tests and a real Qwen3.5-4B demo inside a Slurm
GPU allocation. It uses node-local storage for weights and its environment, and
copies small result artifacts back to the submission directory. It never stops
other jobs. See `docs/SURREY_RUNBOOK.md` and `docs/HANDOFF_SPEC.md` for the measured
environment, evidence and the full experiment handoff.

Surrey job **44314 completed successfully** on an RTX A5000: 89 native score items,
zero failures, 167 tests in that source snapshot, 8.55 GiB peak allocated GPU memory.
The optional shared-prefix acceleration exceeded the 0.01-nat tolerance and is
disabled; uncached teacher forcing and selected/full-logit comparison passed.
See [validation evidence](docs/VALIDATION.md).

The demo uses explicitly synthetic text and geometric images/video to check the
software path. It is not a benchmark of psychological or semantic accuracy.

## Commands

`validate`, `prepare`, `measure`, `calibrate`, `export`, `analyze`, `evaluate`,
`optimize-bridges` all use the same library APIs. Run each command with `--help`.

```bash
omnianchor validate --config configs/studies/toy_en.yaml --samples examples/samples.json
omnianchor measure --config configs/studies/toy_en.yaml --samples examples/samples.json --output runs/example.parquet
```

The walkthrough is in `notebooks/01_measurement_walkthrough.ipynb`; reusable anchor
packs are in `configs/anchors`. Experiment protocol YAMLs are separate from runnable
StudySpec YAMLs.

Local dataset adapters: ValueEval, EmoBank, Chinese EmoBank, OASIS, VIVA, WiC,
DWUG, VATEX and FMAT stored probabilities. Dataset fetching is explicit, labels
remain separate from model input, and original dataset licenses still apply.

The optimized bridge API accepts a local generation callback. The CLI selects
from a pre-scored training pool; it does not silently call an external model.

## Research status

E1–E6 are in progress. Matched FMAT comparisons and the frozen OASIS201 direct
image evaluation have verified results; remaining final evaluations and ablations
are running or queued. The full paper and reproduction release are not complete.
See [current research status](docs/RESEARCH_STATUS.md) for results, limitations, and
[the evidence ledger](research/result-ledger.json) for traceable claims.
