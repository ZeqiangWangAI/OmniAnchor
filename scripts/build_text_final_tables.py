"""Generate separate direct, predictive, network and no-content final-result tables."""
import argparse
import json
from pathlib import Path

import pandas as pd

from build_verified_paper_tables import emit_table
from omnianchor.campaign import append_event, create_run
from omnianchor.io import read_json, write_json
from omnianchor.provenance import file_hash


def cell(row, value="value", lower="ci_lower", upper="ci_upper"):
    return f"{row[value]:.3f} [{row[lower]:.3f}, {row[upper]:.3f}]"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["valueeval", "emobank", "chinese", "network", "output"]:
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--controls", type=Path, nargs=3, required=True, help="ValueEval, English, Chinese order")
    args = parser.parse_args()
    folders = [args.valueeval, args.emobank, args.chinese, args.network, *args.controls]
    for folder in folders:
        events = (folder/"events.jsonl").read_text().splitlines()
        if not events or json.loads(events[-1])["status"] != "completed":
            raise ValueError("Require complete final results.")
    sources = [p for folder in folders for p in folder.glob("*.csv")] + [
        args.valueeval/"direct.json", args.network/"summary.json", *[p/"metrics.json" for p in args.controls]]
    create_run(args.output, {"purpose": "Frozen E2 text and ValueEval E6 results; other studies remain outstanding",
        "source_sha256": {str(p): file_hash(p) for p in sources}, "script_sha256": file_hash(Path(__file__)),
        "rounding": "Correlations/AP and intervals3decimals, q4decimals; no selected best-method emphasis"})
    try:
        v = pd.read_csv(args.valueeval/"metrics.csv")
        direct = read_json(args.valueeval/"direct.json")
        if set(v.n) != {1576} or len(v) != 14:
            raise ValueError("Unexpected full ValueEval population or methods.")
        rows = [[row.method, cell(row), f"{direct[row.method]['within_sample']['mean_instance_ap']:.3f}"]
            for _, row in v[v.kind == "direct"].iterrows()]
        emit_table(args.output, "valueeval-direct", "ValueEval1576 protected materials: direct macroAP [95% source-bootstrap CI], ranking materials within each value. Within-material candidate AP is a distinct descriptive axis.",
            ["Coordinates", "Macro AP [95% CI]", "Within-material AP"], rows)
        rows = [[row.method, cell(row)] for _, row in v[v.kind == "probe"].iterrows()]
        emit_table(args.output, "valueeval-predictive", "ValueEval: frozen train-fitted logistic prediction with development-selected regularization, all1576protected materials. This evaluates features plus predictor, not direct coordinates.",
            ["Features", "Macro AP [95% CI]"], rows)
        comparisons = []
        for name, folder, dimensions, count in [("emobank", args.emobank, ["V", "A", "D"], 1000), ("chinese", args.chinese, ["V", "A"], 528)]:
            frame = pd.read_csv(folder/"metrics.csv")
            if set(frame.n) != {count} or len(frame) != 8*len(dimensions):
                raise ValueError("Unexpected affect population or method coverage.")
            rows = []
            for method in frame.method.drop_duplicates():
                group = frame[frame.method == method].set_index("metric")
                rows.append([method, *[cell(group.loc["spearman_"+dimension]) for dimension in dimensions]])
            emit_table(args.output, name+"-predictive", f"{name}: fixed train-fitted Ridge prediction, {count}protected materials. Spearman correlation [95% source-bootstrap CI]; development chooses regularization, no development refit.",
                ["Features", *dimensions], rows)
            paired = pd.read_csv(folder/"paired-differences.csv")
            paired.insert(0, "study", name)
            comparisons.append(paired)
        paired = pd.read_csv(args.valueeval/"paired-differences.csv")
        paired.insert(0, "study", "valueeval")
        comparisons.append(paired)
        pd.concat(comparisons).to_csv(args.output/"all-frozen-text-paired.csv", index=False)
        summary = read_json(args.network/"summary.json")
        rows = [[method, f"{result['point']['edge_weight_spearman']:.3f} [{result['edge_weight_spearman_ci'][0]:.3f}, {result['edge_weight_spearman_ci'][1]:.3f}]",
            f"{result['point']['top_k_edge_jaccard']:.3f}"] for method, result in summary.items()]
        emit_table(args.output, "valueeval-network", "ValueEval protected concept networks:20matched nodes,190edges,105source groups. Correlation with human-label edge weights and fixed top20-edge overlap; edges are not independent cases.",
            ["Coordinates", "Edge Spearman [95% CI]", "Top20 Jaccard"], rows)
        control_rows = []
        for study, folder in zip(["valueeval", "emobank", "chinese"], args.controls):
            frame = pd.read_csv(folder/"paired-control-differences.csv")
            frame.insert(0, "study", study)
            control_rows.append(frame)
        pd.concat(control_rows).to_csv(args.output/"all-no-content-paired.csv", index=False)
        write_json(args.output/"scope.json", {"direct_predictive_separate": True,
            "control_inference": "Each study and direct/probe control family separate; affect uses MSE reduction because constant-predictor correlations are undefined. Do not compare MSE magnitudes across outcome scales.",
            "network_conclusion": "No human-network recovery claim; source-bootstrap native interval includes zero",
            "remaining": "OASIS12/E3/fullVIVA/VATEX/DWUG/remainingE6/cost/finalpaper"})
        append_event(args.output, "completed")
    except BaseException as exc:
        append_event(args.output, "failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
