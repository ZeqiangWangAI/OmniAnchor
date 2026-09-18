# Command line

`omnianchor <command> --help` lists the options. Every command prints one JSON object on
success; configuration and input errors exit with status 2 and a one-line message; a run
with failed score rows exits with status 1 after writing its artifacts.

## `run`

```bash
omnianchor run --config configs/pipelines/toy_quickstart.yaml
```

Executes a pipeline file (see [configuration.md](configuration.md)): measure, optional
calibration, matrix export and analyses, all into `output_dir` with a `run.json` record.

## `validate`

```bash
omnianchor validate --config configs/studies/qwen35_affect12_en.yaml --samples examples/samples.json
```

Resolves aliases and packs, validates the study and the samples, loads no weights.

## `measure`

```bash
omnianchor measure --config STUDY --samples SAMPLES --output scores.parquet [--cache DIR]
```

Scores every sample x anchor x bridge. With `--cache`, successful candidates are stored by
content hash and replayed exactly on the next run.

## `calibrate`

```bash
omnianchor calibrate --reference-scores ref.parquet --reference-split train \
    --output reference.json [--scores scores.parquet --transformed-output calibrated.parquet]
```

Fits per-coordinate reference statistics (log-mean-probability, mean, SD). Samples marked
test or validation are refused. Optionally appends `reference_log_ratio` and `reference_z`
to a score table.

## `export`

```bash
omnianchor export --scores calibrated.parquet --variant reference_z --missing error --output matrix.npz
```

Averages bridges uniformly into a sample x anchor matrix (`.npz` or `.parquet`).

## `analyze`

```bash
omnianchor analyze cluster --input matrix.npz --k 3 --output clusters.json
omnianchor analyze pca --input matrix.npz --output pca.json
omnianchor analyze groups --input matrix.npz --metadata meta.json --bootstrap 1000 --output groups.json
omnianchor analyze shift --input matrix.npz --metadata meta.json --output shift.json
omnianchor analyze network --input matrix.npz --network-kind concept --output network.json
omnianchor analyze reliability --input scores.parquet --output reliability.json
```

`groups` and `shift` read `group_id`, `target_id` and `time` from the matrix manifest or
from `--metadata` (JSON records keyed by `id`). Bootstraps resample source groups.

## `evaluate`

```bash
omnianchor evaluate --kind multilabel|vad|candidates|retrieval --predictions p.csv --labels l.csv --output m.json
```

Rows are aligned by `sample_id` (or `query_id`); mismatched or duplicated ids are errors.

## `prepare`

```bash
omnianchor prepare --config dataset.yaml --output data/prepared/valueeval
```

Converts a locally acquired public dataset (ValueEval, EmoBank, Chinese EmoBank, OASIS,
VIVA, WiC, DWUG, VaTeX, FMAT) into samples, labels and a manifest with file hashes. The
adapters never download data.

## `optimize-bridges`

Bounded subset selection from an already scored training pool of bridge wordings; see
`--help`. It does not call a model.

## `models`

Prints the alias registry and the adapter table.

## `verify-model`

```bash
omnianchor verify-model --config STUDY --output verification.json [--image picture.png]
```

Loads the study's checkpoint on CUDA and runs the native acceptance check described in
[models.md](models.md).
