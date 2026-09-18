import sys

import numpy as np

from scripts.build_development_features import main
from omnianchor.io import read_json, write_json
from omnianchor.types import Part, Sample


def test_image_baselines_assemble_without_imputing_text_only_e5(tmp_path, monkeypatch):
    shards, run, output = tmp_path/"shards", tmp_path/"run", tmp_path/"features"
    shards.mkdir()
    run.mkdir()
    samples = [Sample(id=i, parts=(Part(type="image", path="fixture.png"),),
                      metadata={"split": "dev"}) for i in ["a", "b"]]
    write_json(shards/"samples.json", samples)
    write_json(shards/"manifest.json", {"files": [{"file": "samples.json"}]})
    (run/"exit_code.txt").write_text("0\n")
    for method in ["qwen-embedding", "qwen-reranker"]:
        folder = run/method
        folder.mkdir()
        write_json(folder/"manifest.json", {"method": method, "anchor_ids": ["x", "y"]})
        np.savez(folder/"matrix.npz", scores=[[3., 4.], [1., 2.]], sample_ids=["b", "a"],
                 anchor_ids=["x", "y"], features=[[.3, .4], [.1, .2]] if method == "qwen-embedding" else np.empty((0, 0)))
    monkeypatch.setattr(sys, "argv", ["build_development_features.py", "--shards", str(shards),
        "--baseline-runs", str(run), "--methods", "qwen-embedding", "qwen-reranker", "--output", str(output)])
    main()
    assert set(read_json(output/"features.json")) == {"qwen-embedding-anchor", "qwen-embedding-original", "qwen-reranker-anchor"}
    with np.load(output/"qwen-embedding-anchor.npz") as data:
        np.testing.assert_equal(data["values"], [[1., 2.], [3., 4.]])
        assert data["sample_ids"].tolist() == ["a", "b"]
