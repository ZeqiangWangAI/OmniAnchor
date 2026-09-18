# E6 requirement-to-evidence audit

Audit date: 11 September 2026. This maps the original HANDOFF_SPEC E6 requirements;
it does not replace them or establish successful construct validity.

| Requirement | Inspected evidence | Finding |
|---|---|---|
| Signed Pearson concept edges, full matrix | `runs/cpu-valueeval-evidence-20260911-01/runs/analysis-45995/analysis`, `research/concept-network-export-verification-20260911-01.json` | Six methods, 20 common nodes, 190 edges each; full matrices and CSV/GraphML weights match |
| Human-label comparison, fixed density overlap | Same summary and independently verified matrices | All six methods retain edge Spearman and top-20 Jaccard; native .022 and .143 indicate weak agreement |
| Source-bootstrap edge frequency | Per-method `edges.csv` in the same analysis | 1,000 valid draws per method; frequencies agree with integer selection counts and 20 selected edges per draw; source grouping retained |
| Undefined nodes/edges | All-node adjacency exports and common finite-node comparison | Original undefined values are preserved; no missing human usage judgments replaced with zero; node order/export checks are recorded |
| Independent-column negative control, common-row invariance | `paper/generated/network-controls-20260911-01/controls.csv` and hashed source summary | All six conditions retained; maximum common-row error below 8.9e-16; single seeded shuffle is descriptive, not a significance test |
| Sample graphs and human relatedness | `research/dwug-network-export-verification-20260911-01.json`; protected DWUG analysis 46173 | 72 full network exports, 1,415,728 edges across exports; nine aligned partial human matrices; independent pooled-pair metric verification retained |
| Clustering parameters and template/group stability | `paper/generated/e3-emobank-final-20260911-01/sensitivity.md`, `paper/generated/e3-dwug-final-20260911-01/sensitivity.md` | Fixed k=2, seed42, n_init10; 1,000 source draws; English six and DWUG seven instrument summaries; saved template ARIs independently recomputed |
| No 2D separation as validity | Manuscript Sections 5.8 and 6 | Descriptive stability is explicitly separated from human validity; no claim of accurate human clusters |

## Edge-frequency interpretation boundary

The source-bootstrap edge-frequency output above is the concept-graph estimator:
correlations change when source groups are resampled. For the sample graph,
HANDOFF_SPEC defines edges as fixed representation similarities. A particular pair's
weight cannot change by resampling other nodes. Conditional full-graph edge inclusion
is exactly one when both endpoints are present; unconditional inclusion measures
endpoint availability. This is a mathematical boundary, not an empirical stability
result. No additional variable-node sparsification analysis was introduced after
viewing results. The manuscript now states this distinction explicitly.

The explicit sample-edge export is now retained in
`runs/dwug-edge-bootstrap-20260911-01`: 72 networks and 1,415,728 edge rows.
Each row stores endpoint co-presence count, 1,000 draws, unconditional frequency,
and conditional full-edge inclusion (undefined if endpoints never co-occur).
The draw order follows the existing seed42 within-period source resampling.
This post-result completion export does not add a confirmatory inference.
`research/dwug-edge-bootstrap-verification-20260911-01.json` records coverage,
frequency arithmetic and identical endpoint frequencies across the eight methods.
The overall goal still requires final manuscript and reproduction acceptance.
