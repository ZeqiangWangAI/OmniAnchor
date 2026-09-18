import sys

import numpy as np
import pandas as pd

from scripts.analyze_development_networks import main
from vlanchor.io import read_json, write_json
from vlanchor.types import Part, Sample


def test_network_retains_undefined_edges_without_zero_imputation(tmp_path, monkeypatch):
    dev, features, output = tmp_path/"dev", tmp_path/"features", tmp_path/"result"
    dev.mkdir()
    features.mkdir()
    ids = [str(i) for i in range(40)]
    samples = [Sample(id=i, group_id=i, parts=(Part(type="text", text="fixture"),),
                      metadata={"split": "dev"}) for i in ids]
    write_json(dev/"samples.json", samples)
    rng = np.random.default_rng(42)
    values = np.column_stack([rng.normal(size=(40, 3)), np.ones(40)])
    columns = ["a", "b", "c", "constant"]
    pd.DataFrame(values, index=pd.Index(ids, name="sample_id"), columns=columns).to_csv(dev/"labels.csv")
    write_json(features/"features.json", {"native-raw_logp": {"direct_anchor_scores": True}})
    np.savez(features/"native-raw_logp.npz", values=values, sample_ids=ids, column_ids=columns)
    monkeypatch.setattr(sys, "argv", ["network", "--features", str(features), "--dev", str(dev), "--output", str(output)])
    main()
    full = pd.read_csv(output/"native-raw_logp-all-nodes-adjacency.csv", index_col=0)
    assert full.shape == (4, 4)
    assert full.loc["constant"].isna().all()
    assert full["constant"].isna().all()
    assert read_json(output/"manifest.json")["removed_constant_nodes"] == ["constant"]
    assert read_json(output/"summary.json")["native-raw_logp"]["point"]["edge_weight_spearman"] == 1.
