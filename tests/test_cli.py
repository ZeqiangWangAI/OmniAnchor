"""Offline CLI acceptance tests. Synthetic scores never establish scientific validity."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from omnianchor.cli import main
from omnianchor.io import load_calibration, load_matrix, load_samples, load_scores, read_json, write_json
from omnianchor.types import Anchor, Bridge, ModelSpec, Part, Sample, StudySpec


def command(capsys, *args, expected_exit=0):
    assert main([str(arg) for arg in args]) == expected_exit
    return json.loads(capsys.readouterr().out)


def write_study(tmp_path, *, split="train", n_samples=6, n_bridges=3):
    spec = StudySpec(
        model=ModelSpec(backend="toy", id="synthetic", revision="fixture", device="cpu", precision="fp32"),
        anchors=(Anchor(id="care", surface="care"), Anchor(id="power", surface="power")),
        bridges=tuple(Bridge(id=f"b{i}", prefix=f"Associated concept, wording {i}:\n") for i in range(n_bridges)),
    )
    config = tmp_path / "study.yaml"
    config.write_text(yaml.safe_dump(spec.model_dump(mode="json")), encoding="utf-8")
    samples_path = tmp_path / "samples.json"
    samples = [Sample(id=f"s{i}", parts=(Part(type="text", text=f"Synthetic sample {i}."),),
                      group_id="a" if i < n_samples // 2 else "b", time="old" if i % 2 else "new",
                      metadata={"split": split, "target_id": "word"}) for i in range(n_samples)]
    write_json(samples_path, samples)
    return config, samples_path


def test_cli_validate_measure_calibrate_export_and_analyze(tmp_path, capsys):
    config, samples = write_study(tmp_path)
    valid = command(capsys, "validate", "--config", config, "--samples", samples)
    assert valid["valid"] and not valid["model_weights_loaded"]
    scores, cache = tmp_path / "scores.parquet", tmp_path / "cache"
    result = command(capsys, "measure", "--config", config, "--samples", samples,
                     "--output", scores, "--cache", cache)
    assert result["failed_items"] == 0 and result["score_items"] == 36
    assert load_scores(scores).manifest["model"]["scientific_validity"] is False
    replay = command(capsys, "measure", "--config", config, "--samples", samples,
                     "--output", scores, "--cache", cache)
    assert replay["cache_hits"] == 36
    reference, transformed = tmp_path / "reference.json", tmp_path / "transformed.parquet"
    command(capsys, "calibrate", "--reference-scores", scores, "--reference-split", "train",
            "--output", reference, "--scores", scores, "--transformed-output", transformed)
    assert load_calibration(reference).statistics.std_logp.gt(0).all()
    assert np.isfinite(load_scores(transformed).frame.reference_z).all()
    for extension in (".npz", ".parquet"):
        output = tmp_path / ("matrix" + extension)
        command(capsys, "export", "--scores", scores, "--calibration", reference,
                "--variant", "reference_z", "--output", output)
        assert load_matrix(output).values.shape == (6, 2)
    matrix = tmp_path / "matrix.npz"
    for kind, extra in (("pca", ()), ("cluster", ("--k", "2")),
                        ("network", ()), ("network", ("--network-kind", "sample", "--k", "2")),
                        ("groups", ("--bootstrap", "0")), ("shift", ("--bootstrap", "0"))):
        output = tmp_path / f"analysis-{kind}-{len(extra)}.json"
        command(capsys, "analyze", kind, "--input", matrix, "--output", output, *extra)
        assert read_json(output)
    reliability = tmp_path / "reliability.json"
    command(capsys, "analyze", "reliability", "--input", scores, "--output", reliability)
    assert read_json(reliability)


def test_cli_prepare_preserves_named_label_index_and_resolves_relative_path(tmp_path, capsys):
    csv = tmp_path / "emobank.csv"
    pd.DataFrame({"id": ["x", "y"], "split": ["train", "test"], "V": [2, 4],
                  "A": [3, 4], "D": [4, 2], "text": ["first", "second"]}).to_csv(csv, index=False)
    cfg = tmp_path / "prepare.yaml"
    cfg.write_text(yaml.safe_dump({"dataset": "emobank", "adapter_kwargs": {"path": "emobank.csv"}}))
    output = tmp_path / "prepared"
    result = command(capsys, "prepare", "--config", cfg, "--output", output)
    assert result["samples"] == 2
    assert list(pd.read_csv(output / "labels.csv").columns) == ["sample_id", "V", "A", "D"]
    assert load_samples(output / "samples.json")[1].metadata["split"] == "test"
    assert read_json(output / "dataset.manifest.json")["rating_perspective"] == "combined"


def test_cli_viva_prepare_multiindex_and_candidate_evaluation(tmp_path, capsys):
    annotation = tmp_path / "viva.json"
    write_json(annotation, [{"index": 1, "image_file": "1.jpg", "answer": "B",
                             "action_list": ["A. Leave.", "B. Help."],
                             "reason": "GOLD REASON",
                             "values": {"positive": ["Care: GOLD EXPLANATION"],
                                        "negative": ["Power: IRRELEVANT EXPLANATION"]}}])
    cfg = tmp_path / "viva.yaml"
    cfg.write_text(yaml.safe_dump({"dataset": "viva", "adapter_kwargs": {
        "annotation_path": "viva.json", "media_root": "images", "require_media": False}}))
    output = tmp_path / "prepared"
    command(capsys, "prepare", "--config", cfg, "--output", output)
    labels = pd.read_csv(output / "labels.csv")
    assert list(labels.columns) == ["sample_id", "anchor_id", "relevant"]
    assert load_samples(output / "samples.json")[0].parts[0].path == str(tmp_path / "images" / "1.jpg")
    predictions = labels.rename(columns={"relevant": "score"}).iloc[::-1]
    prediction_path = tmp_path / "candidate-scores.csv"
    predictions.to_csv(prediction_path, index=False)
    metrics = tmp_path / "candidate-metrics.json"
    command(capsys, "evaluate", "--kind", "candidates", "--predictions", prediction_path,
            "--labels", output / "labels.csv", "--output", metrics)
    assert read_json(metrics)["mean_instance_ap"] == 1
    assert read_json(metrics)["mrr"] == 1


@pytest.mark.parametrize("kind,columns", [("multilabel", ["care", "power"]), ("vad", ["V", "A"])])
def test_cli_evaluation_aligns_sample_and_column_ids(tmp_path, capsys, kind, columns):
    labels = pd.DataFrame({"sample_id": ["a", "b", "c"], columns[0]: [0, 1, 1], columns[1]: [1, 0, 1]})
    pred = labels.iloc[::-1][["sample_id", columns[1], columns[0]]]
    lp, pp, output = tmp_path / "labels.csv", tmp_path / "pred.csv", tmp_path / "metrics.json"
    labels.to_csv(lp, index=False)
    pred.to_csv(pp, index=False)
    command(capsys, "evaluate", "--kind", kind, "--predictions", pp, "--labels", lp, "--output", output)
    result = read_json(output)
    assert result["macro_ap" if kind == "multilabel" else "mean_spearman"] == 1


def test_cli_retrieval_aligns_named_candidate_columns(tmp_path, capsys):
    pp, lp, output = tmp_path / "pred.csv", tmp_path / "labels.csv", tmp_path / "retrieval.json"
    pd.DataFrame({"d1": [0.9], "d2": [0.1]}).to_csv(pp, index=False)
    pd.DataFrame({"d2": [0], "d1": [1]}).to_csv(lp, index=False)
    command(capsys, "evaluate", "--kind", "retrieval", "--predictions", pp, "--labels", lp, "--output", output)
    assert read_json(output)["success@1"] == 1


def test_cli_precomputed_bridge_optimization_runs_without_operator(tmp_path, capsys):
    config, samples = write_study(tmp_path, n_bridges=4)
    scores = tmp_path / "scores.parquet"
    command(capsys, "measure", "--config", config, "--samples", samples, "--output", scores)
    labels = tmp_path / "labels.csv"
    pd.DataFrame({"sample_id": [f"s{i}" for i in range(6)], "care": [0, 1, 0, 1, 0, 1],
                  "power": [1, 0, 1, 0, 1, 0]}).to_csv(labels, index=False)
    output = tmp_path / "optimized.json"
    result = command(capsys, "optimize-bridges", "--scores", scores, "--labels", labels,
                     "--variant", "raw_logp", "--split", "train", "--min-class-count", "1", "--output", output)
    artifact = read_json(output)
    assert result["status"] == "ok"
    assert len(artifact["bridges"]) == 3
    assert artifact["manifest"]["scored_bridge_count"] == 4
    assert len(artifact["trace"]) == 1


def test_cli_rejects_test_reference_even_with_explicit_train_override(tmp_path, capsys):
    config, samples = write_study(tmp_path, split="test")
    scores = tmp_path / "test.parquet"
    command(capsys, "measure", "--config", config, "--samples", samples, "--output", scores)
    with pytest.raises(SystemExit) as exc:
        main(["calibrate", "--reference-scores", str(scores), "--reference-split", "train",
              "--output", str(tmp_path / "reference.json")])
    assert exc.value.code == 2
    assert "never test" in capsys.readouterr().err
    assert not (tmp_path / "reference.json").exists()


def test_cli_failed_measurement_has_nonzero_exit_and_explicit_artifact_status(tmp_path, capsys):
    config, samples = write_study(tmp_path)
    payload = yaml.safe_load(config.read_text())
    payload["resources"]["limits"]["input_text_tokens"] = 1
    config.write_text(yaml.safe_dump(payload))
    output = tmp_path / "failed.parquet"
    result = command(capsys, "measure", "--config", config, "--samples", samples,
                     "--output", output, expected_exit=1)
    assert result["failed_items"] == result["score_items"] == 36
    assert set(load_scores(output).frame.status) == {"BudgetExceeded"}
    assert load_scores(output).frame.raw_logp.isna().all()


def test_cli_calibration_requires_both_transform_output_arguments(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["calibrate", "--reference-scores", "unused.parquet", "--output", "unused.json",
              "--scores", "unused-scores.parquet"])
    assert exc.value.code == 2
    assert "supplied together" in capsys.readouterr().err


def test_cli_optimizer_reports_missing_label_columns_as_user_error(tmp_path, capsys):
    config, samples = write_study(tmp_path)
    scores = tmp_path / "scores.parquet"
    command(capsys, "measure", "--config", config, "--samples", samples, "--output", scores)
    labels = tmp_path / "bad-labels.csv"
    pd.DataFrame({"sample_id": [f"s{i}" for i in range(6)], "care": [0, 1, 0, 1, 0, 1]}).to_csv(labels, index=False)
    with pytest.raises(SystemExit) as exc:
        main(["optimize-bridges", "--scores", str(scores), "--labels", str(labels), "--variant", "raw_logp",
              "--split", "train", "--output", str(tmp_path / "artifact.json")])
    assert exc.value.code == 2
    assert "Traceback" not in capsys.readouterr().err
