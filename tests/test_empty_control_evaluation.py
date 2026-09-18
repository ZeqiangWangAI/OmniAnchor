import sys

import numpy as np
import pandas as pd

from scripts.evaluate_empty_controls import main
from omnianchor.io import read_json, write_json
from omnianchor.provenance import file_hash
from omnianchor.types import Part, Sample


def test_constant_no_content_predictor_has_undefined_correlation(tmp_path, monkeypatch):
    data, empty, final, output = [tmp_path/name for name in ["data", "empty", "final", "output"]]
    for folder in [data, empty, final]:
        folder.mkdir()
    ids, columns = [f"i{i}" for i in range(40)], ["V", "A"]
    samples = [Sample(id=i, group_id=str(n//2), parts=(Part(type="text", text="software fixture"),),
        metadata={"split": "test"}) for n, i in enumerate(ids)]
    write_json(data/"samples.json", samples)
    gold = np.random.default_rng(42).normal(size=(40, 2))+[2, 3]
    pd.DataFrame(gold, index=pd.Index(ids, name="sample_id"), columns=columns).to_csv(data/"labels.csv")
    cpath = tmp_path/"scoring.json"
    write_json(cpath, {"status": "frozen", "sample_ids": ids, "label_columns": columns, "task": "regression",
        "samples_sha256": file_hash(data/"samples.json"), "labels_sha256": file_hash(data/"labels.csv")})
    name = "native-raw_logp"
    write_json(empty/"features.json", {name: {"direct_anchor_scores": True}})
    np.savez(empty/f"{name}.npz", features=[[1., 1.]], predictions=[[2., 3.]], column_ids=["a", "b"])
    np.savez(final/f"{name}-predictions.npz", values=gold, sample_ids=ids, label_columns=columns)
    write_json(final/"manifest.json", {"contract_sha256": file_hash(cpath), "script_sha256": "fixture-only"})
    (final/"events.jsonl").write_text('{"status":"completed"}\n')
    contract = tmp_path/"control.json"
    write_json(contract, {"status": "frozen", "analyzer_sha256": file_hash("scripts/evaluate_empty_controls.py"),
        "scoring_contract": str(cpath), "scoring_contract_sha256": file_hash(cpath), "statistics_source_sha256": {},
        "empty_files_sha256": {p.name: file_hash(p) for p in empty.iterdir()}, "final_evaluator_sha256": "fixture-only"})
    monkeypatch.setattr(sys, "argv", ["empty-controls", "--contract", str(contract), "--data", str(data),
        "--empty", str(empty), "--final-results", str(final), "--output", str(output)])
    main()
    result = read_json(output/"metrics.json")[name]
    assert result["no_content_probe"]["spearman"] == {}
    assert result["no_content_probe"]["undefined_dimensions"] == ["V", "A"]
    assert result["actual_mse"] < 1e-25
    np.testing.assert_allclose(result["no_content_mse"], np.mean((gold-[2, 3])**2))
