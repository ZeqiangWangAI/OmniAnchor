"""Pipeline YAML runs the documented chain with the toy backend; synthetic scores prove nothing."""

import json

import numpy as np
import pytest
import yaml

from omnianchor.cli import main
from omnianchor.io import load_calibration, load_matrix, load_scores, read_json, write_json
from omnianchor.pipeline import load_pipeline, run_pipeline
from omnianchor.types import Part, Sample


def write_yaml(path, value):
    path.write_text(yaml.safe_dump(value, allow_unicode=True), encoding="utf-8")
    return path


def fixture(tmp_path, *, split_train=True):
    packs = tmp_path / "packs"
    packs.mkdir(parents=True)
    write_yaml(packs / "anchors.yaml", {"anchors": [{"id": "care", "surface": "care"},
                                                    {"id": "power", "surface": "power"}]})
    write_yaml(packs / "bridges.yaml", {"bridges": [
        {"id": "b1", "prefix": "Associated concept, wording 1:\n"},
        {"id": "b2", "prefix": "Associated concept, wording 2:\n"}]})
    write_yaml(tmp_path / "study.yaml", {"model": "toy", "anchors": "packs/anchors.yaml",
                                         "bridges": "packs/bridges.yaml"})
    samples = [Sample(id=f"s{i}", parts=(Part(type="text", text=f"Synthetic sample {i}."),),
                      group_id="a" if i < 3 else "b", time="old" if i % 2 else "new",
                      metadata=({"split": "train" if i < 4 else "test"} if split_train else {})
                      | {"target_id": "word"})
               for i in range(6)]
    write_json(tmp_path / "samples.json", samples)
    return tmp_path


def test_pipeline_runs_measure_calibrate_export_and_analyses(tmp_path):
    root = fixture(tmp_path)
    pipeline = write_yaml(root / "pipeline.yaml", {
        "study": "study.yaml", "samples": "samples.json", "output_dir": "runs/demo",
        "cache": "runs/cache", "reference": {"split": "train"},
        "export": {"variant": "reference_z"},
        "analyses": [{"kind": "pca"}, {"kind": "cluster", "k": 2}, {"kind": "network"},
                     {"kind": "network", "network_kind": "sample", "k": 2},
                     {"kind": "groups", "bootstrap": 20}, {"kind": "shift", "bootstrap": 20},
                     {"kind": "reliability"}]})
    result = run_pipeline(pipeline)
    out = root / "runs/demo"
    assert result["failed_items"] == 0 and result["score_items"] == 24
    assert load_scores(out / "scores.parquet").manifest["model"]["scientific_validity"] is False
    reference = load_calibration(out / "reference.json")
    assert reference.manifest["reference_sample_ids"] == ["s0", "s1", "s2", "s3"]
    assert reference.manifest["reference_splits"] == ["train"]
    matrix = load_matrix(out / "matrix.npz")
    assert matrix.values.shape == (6, 2) and np.isfinite(matrix.values).all()
    for name in ("pca", "cluster", "network-concept", "network-sample", "groups", "shift",
                 "reliability"):
        assert (out / f"analysis-{name}.json").is_file(), name
    record = read_json(out / "run.json")
    assert record["reference"]["split"] == "train" and record["matrix"]["variant"] == "reference_z"
    assert record["study"]["anchors"][0]["id"] == "care"
    replay = run_pipeline(pipeline)
    assert replay["cache_hits"] == 24


def test_pipeline_without_reference_exports_raw_and_inline_study(tmp_path):
    root = fixture(tmp_path)
    study = yaml.safe_load((root / "study.yaml").read_text())
    pipeline = write_yaml(root / "pipeline.yaml", {
        "study": study, "samples": "samples.json", "output_dir": "out",
        "export": {"variant": "raw_logp"}})
    result = run_pipeline(pipeline)
    assert result["artifacts"] == {"scores": "scores.parquet", "matrix": "matrix.npz"}
    assert load_matrix(root / "out/matrix.npz").variant == "raw_logp"


def test_pipeline_validation_errors_name_the_section(tmp_path):
    root = fixture(tmp_path)
    base = {"study": "study.yaml", "samples": "samples.json", "output_dir": "out"}
    cases = [
        ({**base, "export": {"variant": "reference_z"}}, "requires a reference"),
        ({**base, "reference": {"split": "train", "scores": "x.parquet"}}, "exactly one"),
        ({**base, "reference": {"split": "train"}, "analyses": [{"kind": "cluster"}]}, "explicit k"),
    ]
    for raw, message in cases:
        with pytest.raises(ValueError, match=message):
            load_pipeline(write_yaml(root / "pipeline.yaml", raw))
    with pytest.raises(ValueError, match="extra"):
        load_pipeline(write_yaml(root / "pipeline.yaml", {**base, "unknown": 1}))


def test_reference_split_train_requires_marked_samples_and_external_rejects_declared(tmp_path):
    root = fixture(tmp_path, split_train=False)
    with pytest.raises(ValueError, match="metadata.split=train"):
        run_pipeline(write_yaml(root / "p.yaml", {"study": "study.yaml", "samples": "samples.json",
                                                    "output_dir": "out", "reference": {"split": "train"}}))
    result = run_pipeline(write_yaml(root / "p.yaml", {
        "study": "study.yaml", "samples": "samples.json", "output_dir": "out",
        "reference": {"split": "external"}}))
    assert read_json(root / "out/run.json")["reference"]["note"]
    assert result["failed_items"] == 0
    declared = fixture(tmp_path / "declared")
    with pytest.raises(ValueError, match="declare splits"):
        run_pipeline(write_yaml(declared / "p.yaml", {
            "study": "study.yaml", "samples": "samples.json", "output_dir": "out",
            "reference": {"split": "external"}}))


def test_external_reference_samples_and_precomputed_scores(tmp_path):
    root = fixture(tmp_path)
    external = [Sample(id=f"r{i}", parts=(Part(type="text", text=f"Reference {i}."),),
                       metadata={"split": "external"}) for i in range(4)]
    write_json(root / "reference.json", external)
    first = write_yaml(root / "p1.yaml", {"study": "study.yaml", "samples": "samples.json",
                                          "output_dir": "one", "reference": {"samples": "reference.json"}})
    run_pipeline(first)
    artifact = load_calibration(root / "one/reference.json")
    assert artifact.manifest["reference_splits"] == ["external"]
    second = write_yaml(root / "p2.yaml", {"study": "study.yaml", "samples": "samples.json",
                                           "output_dir": "two",
                                           "reference": {"scores": "one/scores.parquet"}})
    with pytest.raises(ValueError, match="never test"):
        run_pipeline(second)


def test_cli_run_and_models(tmp_path, capsys):
    root = fixture(tmp_path)
    pipeline = write_yaml(root / "pipeline.yaml", {
        "study": "study.yaml", "samples": "samples.json", "output_dir": "out",
        "reference": {"split": "train"}, "analyses": [{"kind": "pca"}]})
    assert main(["run", "--config", str(pipeline)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["failed_items"] == 0 and "analysis-pca" in printed["artifacts"]
    assert main(["models"]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert listing["aliases"]["qwen3.5-4b"]["adapter"] == "qwen3_5"
    assert "causal_lm" in listing["adapters"]
