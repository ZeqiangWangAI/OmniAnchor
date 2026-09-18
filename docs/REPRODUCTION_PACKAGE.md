# Reproduction package assembly status

This is an assembly checklist, not a claim that a complete distributable archive exists.
The scientific scope follows HANDOFF_SPEC with the 2026-09-11 video amendment.
No protected result is used to select another method or restart optimization.

E1's existing outputs are indexed with hashes in
`research/fmat-output-map-20260911-01.json`: aligned occupation/name scores,
template scores, signed correlations, independent R verification, and masked-model
token audits. The full-sentence PLC audits contain 400 occupation and 400 name
rows; prefix + anchor + suffix equals the saved full-sentence log probability in
all 800 rows (maximum recorded discrepancy zero). This decomposition does not
make PLC equivalent to anchor-only measurement. Retain the original filenames
and map them to HANDOFF_SPEC deliverables rather than creating ambiguous duplicate
result tables.

| Required component | Authoritative location | Assembly status |
|---|---|---|
| Source, environment and pinned models | Git commits, pyproject.toml, configs/models-20260910.json, per-run environment logs | Clean CUDA rebuild 47467 passed installation and 243 frozen-snapshot tests with all 81 package versions matching production; native/Qwen GPU acceptance 47472 and E5 acceptance 47491 passed; final archive pending |
| Acquisition, terms and split provenance | scripts/, data/prepared/, research/source-discovery/; paper/generated/data-acquisition-20260911-01/sources.json | Metadata index verifies all 22 listed acquisition files against original hashes; public allowlist and final rights review remain separate |
| Frozen study configurations | research/contracts/ and immutable HPC releases | Original protocols retained; newly submitted final contracts must be mirrored locally |
| Long scores and run manifests | runs/ and Surrey scratch campaign runs | Major final evidence, seven OASIS12 raw runs, DWUG development and final sensitivity/reference outputs mirrored with individual hashes verified |
| Calibration and score matrices | Per-study feature/analysis folders | Preserve complete arrays; do not substitute paper summaries |
| Networks and statistical outputs | Final ValueEval and DWUG analyses; docs/E6_ACCEPTANCE.md | Full matrices, template/source-cluster stability and explicit sample-edge endpoint frequencies retained; package assembly remains |
| Search history and failures | Campaign search runs, research/contracts/, retry records, Slurm logs | All nine search histories indexed; final failure table covers all 22 failed allocations; include original traces and logs in private evidence package |
| Tables and figure provenance | paper/generated/, paper/claim-evidence-20260911-01.json | Partial; every final table needs source and code hashes |
| Paper and references | paper/manuscript.md, references.bib, submission-materials.md | Incomplete; final results and review still required |
| Reproduction commands | scripts/hpc/ launchers and analysis/plot scripts | Exact launchers retained; clean-environment end-to-end instructions pending |

## Distribution boundary

The complete local evidence snapshot is selected by
`research/private-evidence-allowlist-20260911-01.json`: 208,000 files and
6,114,076,984 uncompressed bytes. It includes the tracked files at commit
`a70b42b`, plus local data and run artifacts, excluding generated Python caches.
It deliberately retains earlier failed outputs and duplicate historical snapshots.
This is private evidence, not a public release. Later manuscript edits and final
shareable deliverables remain separate from this immutable scientific snapshot.

The archive command is:

```sh
python scripts/package_allowlist.py --allowlist research/private-evidence-allowlist-20260911-01.json --output dist/omnianchor-private-evidence-20260911-01.tgz
```

The builder rechecks every selected file hash before writing. Successful archive
completion and fresh extraction must be verified separately; the allowlist itself
does not prove that either operation finished.

Maintain a private evidence archive separately from any public package. Do not recursively
publish data/, runs/, or HPC source snapshots: manifests and sample files can contain
original copyrighted text, human ratings, images, video, or restricted corpus contexts.

Chinese EmoBank provider terms restrict use to academic research and prohibit transferring
its data to third parties. Exclude original sentences, original human rating rows, and
embedded copies in manifests or logs from a public package. Supply acquisition instructions,
provider citation, version/split verification, and aggregate results where permitted.
The source-term audit is research/source-discovery/primary-literature-verification-20260911-03.json.

DWUG contexts and other third-party media require their own license/terms checks; neither
public availability nor a source URL is sufficient redistribution authorization. Keep
acquisition code and provenance pointers when content cannot be redistributed. Numeric
outputs also require inspection for embedded text/identifiers and applicable source terms.
No archive is labeled legally shareable until its explicit file allowlist is reviewed.

The DWUG EN 3.0.0 landing page at https://zenodo.org/records/14028531 also states
that republication and redistribution are prohibited. This provider note is retained
alongside the earlier API license metadata; the latter alone is not treated as
redistribution authorization. Exclude original usage contexts and human judgment
rows from the public package. The version-specific citation and source check are
in `research/source-discovery/primary-literature-verification-20260911-04.json`.

## Final acceptance

Verify a fresh extraction using documented commands, pinned dependencies and source hashes.
Recompute representative numeric results independently; retain undefined values and failed
runs. Reconcile all E1–E6 requirements, the nine figure classes, protected sample coverage,
and reported claims against actual artifacts. Pending runs or missing evidence keep the
package and the system goal incomplete. Actual external submission is outside authorization.

## Rechecked text preparation commands

The following commands were executed successfully against preserved raw inputs in
fresh output directories. Use new output paths on each run; existing preparation
and failed attempts must remain intact.

```sh
python scripts/prepare_text_smoke.py --raw data/raw/text-20260910-01 --output NEW_TEXT_PREPARATION
python scripts/prepare_valueeval.py --raw data/raw/valueeval-hf-1259eea6 --output NEW_VALUEEVAL_PREPARATION
```

The first command recreates complete English reader and Chinese sentence splits,
plus 16 train/dev smoke materials per language. The second uses the acquired
pinned HF Parquet release, not the failed original Zenodo TSV acquisition, and
also requires the preserved dataset README to validate label order. Its actual
group-preserving reference/search sizes are 98/279, not exactly 64/256.
Across both runs, 38 of 39 emitted files are byte-identical to the frozen originals.
The remaining ValueEval manifest differs only in six paths to the newly written
converted TSV files; all other JSON fields match. Evidence:
`research/text-preparation-reproduction-20260911-01.json`.
English retains 131 source groups crossing the official splits; reproduction does
not remove them or transform this evaluation into an independent-source design.
These commands rebuild data only and do not perform model inference or constitute
an all-study replication. The acquisition metadata index contains URLs and hashes
for 22 acquired source files; additional image downloads and dataset README files
are outside that narrow index and must remain in the complete provenance inventory.

## ValueEval network export audit (11 September 2026)

All six concept-network methods pass the saved-export audit: 20 identically ordered
nodes, all 190 undirected edges, matching CSV/GraphML weights and interval attributes,
and top-20 selection frequencies consistent with 1,000 valid source-bootstrap draws.
Edge-weight Spearman correlations were independently recomputed from saved matrices.
Evidence: `research/concept-network-export-verification-20260911-01.json`; command:

```sh
python scripts/verify_concept_network_exports.py --analysis runs/cpu-valueeval-evidence-20260911-01/runs/analysis-45995/analysis --output NEW_AUDIT_DIRECTORY
```

This checks export integrity and saved-matrix comparisons, not independent
reconstruction of every bootstrap draw or successful construct validity. DWUG
sample-network and clustering requirements remain separately tracked.

## DWUG complete network exports (11 September 2026)

`runs/dwug-network-exports-20260911-01` supplies edge CSV and full GraphML for
all 72 method-by-target networks (1,415,728 undirected edges across exports).
The nine human-reference NPZ matrices use the identical node order across methods;
unrated pairs and diagonals remain NaN, with original ordinal judgments retained.
All exported weights were checked against original adjacency arrays, and all observed
human ratings against their aligned matrices. The audit is
`research/dwug-network-export-verification-20260911-01.json`.

```sh
python scripts/export_dwug_networks.py --analysis runs/final-results-update-20260911-01/runs/analysis-46173/analysis --output NEW_EXPORT_DIRECTORY
```

These are private evidence exports pending distribution review. This completes format
coverage, not outstanding bootstrap/cluster-stability requirements.

## Bridge search history inventory

`research/bridge-search-history-index-20260911-01.json` indexes all nine locally
preserved selected artifacts, their exact bridges, generation traces, event counts
and source hashes. Five searches completed; four earlier attempts retained their
`insufficient_valid_bridges` status. Completed reference-guided, random, validity,
reliability and alpha searches retained 9, 11, 9, 12 and 12 candidate directories,
respectively. The latter four reused eight common seeds. Equal configured upper
bounds do not imply equal realized scoring cost. This inventory does not replace
the remaining independent candidate-score and training-membership audit.

The independent input audit in
`research/bridge-training-membership-verification-20260911-01.json` verifies exact
reference/search content against prepared official training records, matching frozen
input hashes and all 106 available raw-score manifests. Reference and search source
groups are disjoint. Failed attempts have no raw-score manifests and are not
counted as successful measurements. Optimizer-objective recomputation remains a
separate check. Reproduce with:

```sh
python scripts/verify_bridge_training_membership.py --output NEW_AUDIT_JSON
```

`research/bridge-objective-verification-20260911-01.json` independently recomputes
all 15 saved generation-level selected-subset objectives from training matrices.
Threshold-grouped AP, rank correlations, and within-anchor alpha reproduce the
saved validity, reliability and selection objectives within 1e-12. This audit does
not establish optimality against every unselected subset or reconstruct generation
randomness. Command:

```sh
python scripts/verify_bridge_objectives.py --output NEW_OBJECTIVE_AUDIT_JSON
```

## Distinct network bootstrap estimands

`research/dwug-network-bootstrap-scope-20260911-01.json` records why the
ValueEval concept-edge bootstrap cannot be interpreted as a DWUG sample-edge
bootstrap. With fixed vectors and calibration, a particular usage-pair cosine
is unchanged by resampling other source documents. In a complete graph, conditional
edge inclusion is one whenever both endpoints are present; unconditional inclusion
measures endpoint availability. Neither is additional semantic-validity evidence.
Template and source-bootstrap clustering stability must be reported separately.
This scope audit does not mark the remaining E6 acceptance complete.

## Private source extraction check

A 252-file source snapshot was archived and freshly extracted with every file hash
verified. All 250 tests passed against its extracted source using the existing local
analysis environment (6.49 s). Evidence and the exact command are in
`research/source-extraction-verification-20260911-01.json`; the private archive is
`runs/source-extraction-check-20260911-01/private-source-snapshot.tgz`. This does not
replace the separate clean-dependency/GPU acceptance or approve public redistribution.
The complete results and final paper package remain pending.

## Cost and failed-allocation evidence

`research/campaign-cost-update-20260911-04.json` records 278 allocations and
146.9689 observed GPU hours, including three still-running allocations at collection.
All five downloaded snapshot files matched remote SHA256 values. Installation job
47467 has no GPU allocation; GPU acceptance jobs 47472 and 47491 are included.
These are provisional observed costs, not the final campaign total.

`paper/generated/failure-index-20260911-02/documented-failures.csv` covers all 22
failed allocations in that snapshot. Nine have structured failure events; the other
13 have Slurm log exceptions. `research/failure-array-log-mapping-20260911-01.json`
resolves raw job IDs to array-index log filenames; retain this mapping so missing
raw-ID filenames are not mistaken for missing logs. The table's exact allocation
coverage, exit codes, elapsed times and costs were verified in
`research/failure-table-verification-20260911-02.json`. Keep pending video withdrawals
and the separately preserved local transfer failure outside this allocation table.
Refresh accounting after the remaining jobs finish; do not overwrite prior snapshots.

## Frozen DWUG sensitivity references

All five frozen reference runs are mirrored in
`runs/dwug-sensitivity-reference-evidence-20260911-01` (670 files). The archive hash
matches the remote original; see `research/dwug-sensitivity-reference-backup-20260911-01.json`.
`research/dwug-reference-coverage-verification-20260911-01.json` checks 64 unique
train-marked reference inputs per instrument, exact score coverage, zero recorded
failures and disabled shared-prefill. `research/dwug-reference-instrument-verification-20260911-01.json`
checks every reference instrument hash against its frozen final submission.
These checks preserve the original calibration inputs; they do not certify the
still-pending final sensitivity outputs or replace raw-value verification.

## Hash-locked source component (11 September 2026)

The explicit 259-file candidate list is
`research/source-package-allowlist-20260911-01.json`. Build it with:

```sh
python scripts/package_allowlist.py --allowlist research/source-package-allowlist-20260911-01.json --output NEW_SOURCE_ARCHIVE.tgz
```

The builder rejects changed file bytes, paths outside the root, symlinks, and an
existing output archive. It never traverses unlisted directories. The candidate
`dist/omnianchor-source-candidate-20260911-01.tgz` was freshly extracted, all 259
hashes matched, and all 253 tests passed using the existing analysis environment
(6.90 seconds). Evidence: `research/source-package-extraction-verification-20260911-01.json`.
This verifies the source component only. The candidate still needs final embedded
content review, and the full result/figure/manuscript package remains separate.
It does not redistribute datasets or establish a fresh GPU experiment replication.

Final campaign accounting supersedes the provisional snapshot above:
`research/campaign-cost-final-20260911-05.json` records 281 terminal allocations,
259 completed and 22 failed, totaling 148.500833 GPU hours. The unchanged 22 failed
allocations match the previously verified failure index. Pending withdrawals and
two local transfer failures are indexed separately in
`paper/generated/campaign-cost-final-20260911-01/manifest.json`.
