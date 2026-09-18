# Experiment protocol fragments

Task-specific configuration fragments of the manuscript's experiments E1–E6. They contain
no results, do not download data, and are not runnable studies; `schema.json` defines
their structure. Each `inputs` entry maps to `omnianchor.datasets.load_dataset(adapter,
**kwargs)`, which reads locally acquired files, records their SHA256 and split
assignments, and keeps labels separate from model inputs. Replace the `data/...` paths
with your acquired, versioned copies before preparing inputs.

| Experiment | Question | Fragments |
|---|---|---|
| E1 | scoring correctness and the FMAT/PLC connection | `fmat_reproduction.yaml` |
| E2 | external and incremental validity (ValueEval, English/Chinese EmoBank, OASIS) | `valueeval_validity.yaml`, `affect_languages.yaml`, OASIS input in `images_values.yaml` |
| E3 | sensitivity and bridge optimisation | calibration data from `valueeval_validity.yaml` and `affect_languages.yaml` |
| E4 | multimodal information use (VIVA, VaTeX) | VIVA input in `images_values.yaml`, `video_retrieval.yaml` |
| E5 | semantic shift (DWUG, WiC) | `semantic_structure.yaml` |
| E6 | semantic networks and clustering | `valueeval_validity.yaml`, `semantic_structure.yaml` |

Dataset notes: EmoBank reader ratings follow the authors' aggregation notebook
(`rating_perspective='reader'` with `individual_ratings_path`); OASIS expects the image-level
`OASIS.csv` from the stimulus archive with local image files; FMAT imports stored model
probabilities, not human labels. Published train/dev/test assignments are preserved when
passed; otherwise a seeded hash split is recorded in the manifest and never silently
reassigned. Original dataset licenses apply and nothing is redistributed here.
