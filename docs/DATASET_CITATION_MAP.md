# Dataset versions and manuscript citations

This map identifies the acquired versions used by the campaign. It does not
replace the individual preparation manifests, authorize redistribution, or claim
that an upstream download will remain available. Historical acquisition failures
and superseded preparations remain preserved.

| Resource | Acquired version / identity | Manuscript citation | Role |
|---|---|---|---|
| FMAT | OSF files under `data/raw/fmat-20260910`; four file hashes in the acquisition index | Bao (2024), DOI 10.1037/pspa0000396 | Stored masked-model scores and shared external occupational/name criteria |
| ValueEval | HF revision `1259eea6eb37980172311bc02ac5ea88e82b9e42` | Kiesel et al. (2023), DOI 10.18653/v1/2023.semeval-1.313 | Official train/dev/test; acquired Parquet conversion, not byte-identical original TSV files |
| EmoBank | Repository commit `248ce2a43e165a66d31aeaed83cff9641d6654e0` | Buechel and Hahn (2017), ACL E17-2092 | Reader ratings reconstructed using individual-reader inputs and official text/splits |
| Chinese EmoBank | Repository commit `eaf0ca0fd6ec9e48e701231eee490d58bd074c12`; CVAS sentence file | Lee et al. (2022); Yu et al. (2016), DOI 10.18653/v1/N16-1066 | Chinese sentence valence/arousal; CVAT is not combined with CVAS |
| OASIS | OSF stimulus archive `uxvpb`; SHA256 `b1c1f23a2adfb504801ad05df00e01b5530f4442524c983fe535476807137312` | Kurdi et al. (2017), DOI 10.3758/s13428-016-0715-3 | Image-level metadata and 900 images; direct/predictive protocols retain their separate train/reference roles |
| VIVA | HF revision `75e6aed088b1e9fc6175d66a9e02b48e5f801982` | Hu et al. (2024), DOI 10.18653/v1/2024.emnlp-main.137 | Action-conditioned value ranking; exact-image grouping uses preparation03 |
| WiC | Official v1 package; SHA256 `f1a2fb67d903c5b9b1180f1035d4228f7c0254e8f5f868d556235457046bd4b2` | Pilehvar and Camacho-Collados (2019), DOI 10.18653/v1/N19-1128 | Training-only general anchors and external reference contexts |
| DWUG EN | Version 3.0.0; DOI 10.5281/zenodo.14028531; archive SHA256 `64eef477154b82cb27925ab4ea8c030a8e23840b538dd06b6464aa1e55af2dbf` | Schlechtweg et al. (2021), resource paper; Schlechtweg et al. (2024), version-specific dataset | Source-grouped usage-pair, change and network evaluations; target-disjoint protected set |
| VaTeX | Official validation v1.0 JSON; SHA256 `838212d8eead2e22c8838cf58530b94868f1a4905b8322b639212122c8033708` | X. Wang et al. (2019), ICCV | Small video capability demonstrations under the user's amendment; no complete retrieval evaluation |

The URL/file index is `paper/generated/data-acquisition-20260911-01/sources.json`.
Its 22 listed files were checked against preserved original acquisition hashes.
It excludes individual image/video downloads and separately acquired README files;
those are retained in the private acquisition and preparation manifests.
Text/ValueEval fresh preparation is verified in
`research/text-preparation-reproduction-20260911-01.json`.

Dataset citation metadata and model/software citations are rendered from
`paper/references.bib`. The separate checkpoint table is
`configs/models-20260910.json`; a current model-card URL does not replace an
executed checkpoint revision. Dataset-specific provider restrictions are recorded
in `docs/REPRODUCTION_PACKAGE.md` and the primary-source verification records.
No original dataset or media file is included merely because its citation appears
in this map.
