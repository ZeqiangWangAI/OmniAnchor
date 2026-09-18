"""Add explicitly exploratory multiplicity control to every existing FMAT comparison."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from vlanchor.analysis import bh_fdr
from vlanchor.campaign import append_event, create_run
from vlanchor.io import read_json, write_json
from vlanchor.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["original", "pilot-manifest", "contract", "output"]:
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    contract = read_json(args.contract)
    if contract["status"] != "frozen" or contract["script_sha256"] != file_hash(Path(__file__)):
        raise ValueError("Exploratory supplement changed after specification.")
    for name, digest in contract["original_files_sha256"].items():
        if file_hash(args.original/name) != digest:
            raise ValueError("Original FMAT evidence changed.")
    if file_hash(args.pilot_manifest) != contract["pilot_manifest_sha256"]:
        raise ValueError("Original numerical pilot target list changed.")
    paired = pd.read_csv(args.original/"paired-differences.csv")
    if len(paired) != 36 or paired.duplicated(["study", "subset", "primary", "comparator"]).any():
        raise ValueError("Require all original36 comparisons, including pilot-excluded sensitivity.")
    pilot = set(read_json(args.pilot_manifest)["targets"])
    create_run(args.output, {"purpose": contract["purpose"], "contract_sha256": file_hash(args.contract),
        "timing": contract["timing"], "family": "All36original paired Pearson comparisons, BH together",
        "bootstrap": "Same1000pairedtargetdraws,seed42; conditional on original fixed50target normalization",
        "new_model_scores": False})
    try:
        rows, maximum_error = [], 0.
        for (study, subset), comparisons in paired.groupby(["study", "subset"], sort=False):
            matrix = pd.read_csv(args.original/f"{study}-aligned-scores.csv", index_col="target")
            if subset == "exclude8pilot42":
                matrix = matrix.loc[~matrix.index.isin(pilot)]
            elif subset != "all50":
                raise ValueError("Unexpected comparison subset.")
            y = matrix.P_male.to_numpy()
            draws = np.random.default_rng(42).integers(len(y), size=(1000, len(y)))
            estimates = {}
            for name in set(comparisons.primary) | set(comparisons.comparator):
                x = matrix[name].to_numpy()
                estimates[name] = (float(pearsonr(x, y).statistic),
                    np.array([pearsonr(x[i], y[i]).statistic for i in draws]))
            for row in comparisons.to_dict("records"):
                a, da = estimates[row["primary"]]
                b, db = estimates[row["comparator"]]
                delta, point = da-db, a-b
                if not np.isfinite(delta).all():
                    raise ValueError("Undefined original paired statistic; no imputation.")
                lo, hi = np.quantile(delta, [.025, .975])
                error = float(np.max(np.abs(np.array([point, lo, hi])-
                    [row["pearson_difference"], row["ci_lower"], row["ci_upper"]])))
                maximum_error = max(maximum_error, error)
                if error > 1e-12:
                    raise ValueError("Supplement does not independently reproduce the original estimate/interval.")
                rows.append({**row, "centered_bootstrap_p": (1+np.count_nonzero(np.abs(delta-point) >= abs(point)))/1001,
                    "n": len(y), "status": "exploratory_after_original_results_seen"})
        for row, q in zip(rows, bh_fdr([r["centered_bootstrap_p"] for r in rows])):
            row["all36_bh_q"] = float(q)
        pd.DataFrame(rows).to_csv(args.output/"paired-differences.csv", index=False)
        write_json(args.output/"verification.json", {"comparisons": len(rows), "max_original_difference": maximum_error,
            "interpretation": "Post-results multiplicity supplement; never a preregistered confirmatory family"})
        append_event(args.output, "completed")
    except BaseException as exc:
        append_event(args.output, "failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
