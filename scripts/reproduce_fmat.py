"""Signed stored-score FMAT reproduction; these are historical outputs, not new inference."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from omnianchor.campaign import create_run, append_event
from omnianchor.io import write_json
from omnianchor.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = {"purpose": "FMAT stored output statistical reproduction; not new human annotations or new model inference",
                "preselected_studies": ["d1a", "d1b", "d3a"],
                "direction": "log P(Male token) minus log P(Female token); positive = male association",
                "aggregation": "within model/query sample-SD z-score across targets; mean z across queries",
                "statistics": "signed Pearson and Spearman with published male proportion; 1000 target bootstrap, seed42",
                "source_hashes": {p.name: file_hash(p) for p in args.data.glob("*.csv")}}
    create_run(args.output, manifest)
    correlations, scores, pairs = [], [], []
    for study, gold_name, key in [("d1a", "stats.occupation", "job"), ("d1b", "stats.name", "name")]:
        frame = pd.read_csv(args.data / f"{study}.csv")
        if (frame.prob <= 0).any() or (frame.prob > 1).any():
            raise ValueError("Stored probabilities must be positive and at most one for log ratios.")
        frame["logp"] = np.log(frame.prob)
        wide = frame.pivot(index=["model", "query", "T_word", "M_pair"], columns="MASK", values="logp")
        if wide[["Male", "Female"]].isna().any().any():
            raise ValueError("Unpaired male/female stored scores.")
        wide["LPR_male_minus_female"] = wide.Male - wide.Female
        d = wide.reset_index()
        d["target"] = d.T_word.str.replace(r"^(?:a|an) ", "", regex=True) if study == "d1a" else d.T_word
        gold = pd.read_csv(args.data / f"{gold_name}.csv").rename(columns={key: "target"})
        d = d.merge(gold[["target", "P_male"]], on="target", validate="many_to_one")
        if len(d) != len(wide):
            raise ValueError("Gold alignment lost targets.")
        d["z"] = d.groupby(["model", "query"])["LPR_male_minus_female"].transform(lambda x: (x - x.mean()) / x.std(ddof=1))
        d["study"] = study
        scores.append(d)
        averaged = d.groupby(["model", "target"], as_index=False).agg(z=("z", "mean"), P_male=("P_male", "first"))
        for model, subset in averaged.groupby("model", sort=True):
            x, y = subset.z.to_numpy(), subset.P_male.to_numpy()
            rng = np.random.default_rng(42)
            draws = []
            for _ in range(1000):
                indices = rng.integers(0, len(x), len(x))
                draws.append(float(pearsonr(x[indices], y[indices]).statistic))
            correlations.append({"study": study, "model": model, "n_targets": len(x),
                                 "pearson_r": pearsonr(x, y).statistic, "spearman_rho": spearmanr(x, y).statistic,
                                 "pearson_ci_low": np.quantile(draws, .025), "pearson_ci_high": np.quantile(draws, .975)})
    frame = pd.read_csv(args.data / "d3a.csv")
    frame["logp"] = np.log(frame.prob)
    wide = frame.pivot(index=["model", "query", "ATTRIB", "A_pair", "A_word", "M_pair"], columns="MASK", values="logp")
    if wide[["Male", "Female"]].isna().any().any():
        raise ValueError("Unpaired relation scores.")
    wide["LPR_male_minus_female"] = wide.Male - wide.Female
    d = wide.reset_index()
    d.to_csv(args.output / "d3a-signed-scores.csv", index=False)
    for model, subset in d.groupby("model"):
        means = subset.groupby("ATTRIB")["LPR_male_minus_female"].mean()
        pairs.append({"model": model, "career_mean": means["Career"], "family_mean": means["Family"],
                      "career_minus_family": means["Career"] - means["Family"]})
    pd.concat(scores).to_csv(args.output / "E1-stored-scores.csv", index=False)
    pd.DataFrame(correlations).to_csv(args.output / "signed-correlations.csv", index=False)
    pd.DataFrame(pairs).to_csv(args.output / "relation-contrasts.csv", index=False)
    write_json(args.output / "summary.json", {"correlations": correlations, "relation_contrasts": pairs,
                                               "new_model_runs": 0, "status": "stored_score_component_only"})
    append_event(args.output, "completed", correlation_rows=len(correlations))
    print(pd.DataFrame(correlations)[["study", "model", "pearson_r", "spearman_rho"]].to_string(index=False))


if __name__ == "__main__":
    main()
