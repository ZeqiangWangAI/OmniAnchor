These files are task-specific configuration fragments for the accepted E1–E6 research plan; they contain no experiment results. Their filenames and `protocol_id` are semantic names, not paper experiment numbers. `accepted_experiments` explicitly records where each fragment contributes. They are not an automatic download or full experiment runner. `schema.json` defines their JSON-compatible YAML structure. Replace local `data/...` paths with acquired, versioned files before preparing inputs.

The accepted paper experiment numbers have the following meaning. The previous numeric filenames used a different grouping and have been removed to avoid that ambiguity.

| Accepted experiment | Scientific question | Supporting configuration fragments |
| --- | --- | --- |
| E1 | Scoring correctness and the FMAT/PLC connection | `fmat_reproduction.yaml`; synthetic scorer/IO tests |
| E2 | External and incremental validity: ValueEval, English/Chinese EmoBank, OASIS | `valueeval_validity.yaml`, `affect_languages.yaml`, the OASIS input in `images_values.yaml` |
| E3 | Sensitivity and PMPO: fixed/optimized bridges, surface/length, event, model, 64/128/256 samples, duplicate/irrelevant anchors, calibration invariants | `valueeval_validity.yaml` and `affect_languages.yaml` provide calibration data; sensitivity axes and controlled sweeps belong to the full handoff specification |
| E4 | Multimodal information use: VIVA action/image ablations, I/T/IT, VATEX retrieval and frame counts | The VIVA input in `images_values.yaml`, plus `video_retrieval.yaml` |
| E5 | Semantic shift using DWUG and WiC with a fixed anchor bank | `semantic_structure.yaml` |
| E6 | Semantic networks and clustering: ValueEval annotation network and DWUG sample graph | `valueeval_validity.yaml`, `semantic_structure.yaml`; use the analysis/network and clustering APIs |

Each `inputs` entry maps directly to `omnianchor.datasets.load_dataset(adapter, **kwargs)`. That returns `DatasetBundle(samples, labels, manifest, candidates)`. The bundle includes file SHA256 values, original source and license notes, split assignments, and missing-media counts. The SHA256 values establish the exact local artifact, not its authenticity; pass `expected_sha256` to an adapter to verify trusted checksums. Data adapters never download or redistribute upstream datasets.

`samples` contain only inputs. Each sample carries `metadata['split']`. `labels` are separate pandas tables. ValueEval, EmoBank, Chinese EmoBank and OASIS labels use a sample-ID index. VIVA uses a `(sample_id, anchor_id)` index and its candidate banks are per-instance. WiC uses a pair-ID index with two referenced sample IDs. DWUG retains individual pair judgments, including explicit undecidable indicators. VATEX stores video/caption relevance edges and represents video and caption as separate samples. FMAT imports stored model probabilities and must not be mistaken for independent human labels.

EmoBank `emobank.csv` is a combined reader/writer annotation, recorded as `rating_perspective='combined'`. `affect_languages.yaml` explicitly selects reader ratings for E2 via `rating_perspective='reader'` plus `individual_ratings_path`. This follows the author's aggregation notebook: exclude all-1 VAD triples, retain IDs with more than one valid rating, average per ID and round to two decimals, then join official text/splits. A published `reader.csv` can instead be supplied as `ratings_path`. The join records IDs excluded for missing ratings or missing official metadata; it never silently calls combined ratings reader ratings.

Official EmoBank sources for the next experiment session:

- Text, split, and combined ratings: https://raw.githubusercontent.com/JULIELab/EmoBank/master/corpus/emobank.csv
- Individual reader ratings: https://raw.githubusercontent.com/JULIELab/EmoBank/master/corpus/individual_reader_ratings.csv
- Already aggregated reader ratings: https://raw.githubusercontent.com/JULIELab/EmoBank/master/corpus/reader.csv
- Exact aggregation procedure: https://github.com/JULIELab/EmoBank/blob/master/corpus/aggregation.ipynb
- Perspective definitions: https://github.com/JULIELab/EmoBank/blob/master/corpus/README.md

For OASIS, the adapter expects the image-level `OASIS.csv` contained in the original stimulus archive (https://osf.io/download/uxvpb/), with local image files. It does not accept the separate participant-level `OASIS_data.csv` (https://osf.io/download/bv43g/). The archive directory was checked with a bounded remote range request, without downloading its 92 MB body. Confirm the extracted image directory and filenames; use `filename_column` or `filename_template` if the local layout differs. Raw participant-data aggregation is outside this adapter.

Preserve published train/dev/test assignments by passing `split` where the original format lacks a split field. Without one, the default is SHA256 of seed and group identifier with expected 60/20/20 proportions; it is order-independent but does not guarantee exact row counts. Official splits are never silently changed to enforce group separation. Crossings are reported in the manifest. Real evaluation must inspect these crossings and protect test data before fitting calibration, a prompt optimizer, a scaler, or a probe.

Evaluation APIs in `omnianchor.evaluation` are independent of model execution:

- `evaluate_multilabel(labels, scores, anchor_ids=None)` ranks samples within each anchor; returns macro/micro AP.
- `evaluate_candidates(labels_by_sample, scores_by_sample)` ranks each sample's local candidates; returns mean per-instance AP and MRR.
- `evaluate_vad(labels, scores, dimensions=('V','A','D'))` reports Spearman with undefined constant dimensions recorded.
- `retrieval_metrics(relevance, scores, ks=(1,5,10))` distinguishes at-least-one-hit success from fraction-of-relevant-items recall.
- `fit_linear_probe(train_x, train_y, dev_x, dev_y, task=...)` fits preprocessing on train only and uses the exact documented dev grid. Its return object exposes `predict_scores` and the full selection trials.
- `group_bootstrap(statistic_of_row_indices, groups)` resamples intact groups; use a paired method-difference statistic for paired intervals.
- `compare_networks(predicted, reference)` requires identical node order and compares undirected off-diagonal weights.

Optional baselines are explicitly text-only: `HFMeanEmbedding`, `HFCrossEncoder`, and `HFSingleTokenMLM` in `omnianchor.baselines`. Their constructors require model ID and revision. Model loading is lazy and local-only by default; CUDA failures do not trigger CPU fallback. Generic mean pooling is a named baseline, not a reproduction of every embedding model's custom recipe. MLM rejects multi-token anchors instead of silently substituting pseudo-likelihood.

Small fixture tests verify published schemas and evaluation mechanics without downloading models or datasets. They do not establish benchmark accuracy or reproduce a paper's numerical results. Full E1–E6 execution, plots and statistical interpretation belong to the subsequent experiment session.
