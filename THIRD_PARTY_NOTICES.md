# Third-party source and data boundaries

VLanchor's repository license does not relicense third-party model weights, datasets,
images, video, annotations or dependencies.

## Qwen embedding and reranking implementation

The files in `research/upstream/qwen3-vl-embedding-393e297/` originate from
QwenLM/Qwen3-VL-Embedding at commit
`393e2978d27852b0d0230d6994f37f9c15bed73c`:

- `qwen3_vl_embedding.py`
- `qwen3_vl_reranker.py`

The directory retains the upstream Apache License 2.0 text in `LICENSE` and
individual retrieval URLs and SHA256 values in `provenance.json`. An integrity
check on 11 September 2026 found these source and license bytes unchanged from
the recorded upstream copies. Preserve this directory's license and provenance
when packaging the vendored source. VLanchor's wrapper in
`src/vlanchor/official_baselines.py` implements its own strict budget and media
handling; those wrapper choices must not be attributed to unmodified upstream
preprocessing.

## Runtime dependencies and model weights

PyTorch, Transformers, Qwen utilities and other dependencies are installed
separately. Their licenses and notices remain applicable. This repository's
environment records identify versions and source revisions; they do not grant
rights to redistribute model checkpoints. Download pinned checkpoints from their
providers under the corresponding model terms.

## Research data

Dataset acquisition and redistribution are reviewed separately from source-code
licensing. In particular, exclude Chinese EmoBank original sentences and original
rating rows from the public reproduction package under the recorded provider terms.
Do not include raw `data/`, `runs/`, media directories or source snapshots by a
recursive packaging rule: they may embed restricted or third-party material.

`docs/REPRODUCTION_PACKAGE.md` specifies the private/public boundary. A final
shareable package still requires an explicit per-file allowlist and the dataset
terms audit. This notice is not a declaration that every repository artifact is
approved for redistribution.
