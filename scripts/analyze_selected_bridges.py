"""Evaluate all train-selected bridge sets on dev without feeding scores back to search."""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from vlanchor import fit_reference, to_matrix, transform
from vlanchor.analysis import bh_fdr
from vlanchor.campaign import append_event, create_run, select_bridge_coordinates
from vlanchor.evaluation import evaluate_candidates, evaluate_multilabel
from vlanchor.io import load_samples, load_scores, read_json, write_json
from vlanchor.provenance import file_hash
from vlanchor.reliability import cronbach_alpha
from vlanchor.types import Bridge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-evaluation", type=Path)
    args = parser.parse_args()
    bank = read_json(args.bank/"manifest.json")
    samples = load_samples(args.dev/"samples.json")
    final = read_json(args.frozen_evaluation) if args.frozen_evaluation else None
    if final:
        if final["status"] != "frozen" or file_hash(Path(__file__)) != final["analysis_source_sha256"]:
            raise ValueError("Bridge analysis changed after final freeze.")
        for name, digest in final["evaluation_files"].items():
            if file_hash(args.dev/name) != digest:
                raise ValueError("Final bridge evaluation population changed.")
        for name, digest in final["bank_files"].items():
            if file_hash(args.bank/name) != digest:
                raise ValueError("Train-selected bridge bank changed.")
        if file_hash(args.selection) != final["selection_sha256"]:
            raise ValueError("Train-only selection artifact changed.")
        for name, digest in final["statistics_source_sha256"].items():
            if file_hash(Path(__file__).resolve().parents[1]/name) != digest:
                raise ValueError("Bridge statistics changed after freeze.")
    if any(s.metadata["split"] != ("test" if final else "dev") for s in samples):
        raise ValueError("Unadmitted bridge evaluation split.")
    ids = [s.id for s in samples]
    labels = pd.read_csv(args.dev/"labels.csv", index_col="sample_id")
    labels.index = labels.index.astype(str)
    if labels.index.duplicated().any() or set(labels.index) != set(ids):
        raise ValueError("Gold population mismatch.")
    anchors = labels.columns.tolist()
    gold = labels.loc[ids].to_numpy(int)
    selection = read_json(args.selection)
    if selection["status"] != "ok" or selection["manifest"]["split"] != "train":
        raise ValueError("Require the original train-only search artifact.")
    eligible = selection["manifest"]["eligible_anchor_ids"]
    eligible_columns = [anchors.index(a) for a in eligible]
    bridges = {b["id"]: Bridge.model_validate(b) for v in bank["sets"].values() for b in v}
    calibrations = {b: fit_reference(load_scores(args.bank/"reference"/f"{b}.parquet")) for b in bridges}
    variants = ["raw_logp", "reference_log_ratio", "reference_z"]
    values = {b: {v: {} for v in variants} for b in bridges}
    sources = {}
    for root in args.runs:
        if (root/"exit_code.txt").read_text().strip() != "0":
            raise ValueError("Failed/incomplete scoring run.")
        if final and read_json(root/"measurement/manifest.json")["frozen_evaluation"]["sha256"] not in final["admitted_scoring_contract_sha256"]:
            raise ValueError("Scoring run was not admitted by the frozen bridge contracts.")
        for path in sorted((root/"measurement").glob("part-*.parquet")):
            metadata = read_json(path.with_suffix(".parquet.manifest.json"))
            part_ids = [s["id"] for s in metadata["samples"]]
            if not set(part_ids) & set(ids):
                continue
            if not set(part_ids) <= set(ids):
                raise ValueError("Part mixes evaluation and non-evaluation populations.")
            raw = load_scores(path)
            for row in metadata["bridges"]:
                b = row["id"]
                if b not in bridges:
                    raise ValueError("Unexpected unselected bridge.")
                selected = select_bridge_coordinates(raw, [bridges[b]])
                transformed = transform(selected, calibrations[b])
                for variant in variants:
                    matrix = to_matrix(transformed, variant=variant, missing="error")
                    if set(matrix.anchor_ids) != set(anchors):
                        raise ValueError("Anchor coverage changed.")
                    for i, identifier in enumerate(matrix.sample_ids):
                        if identifier in values[b][variant]:
                            raise ValueError("Duplicate bridge/sample score.")
                        values[b][variant][identifier] = matrix.values[i, [matrix.anchor_ids.index(a) for a in anchors]]
            sources[str(path)] = file_hash(path)
    for b in bridges:
        for variant in variants:
            if set(values[b][variant]) != set(ids):
                raise ValueError(f"Incomplete frozen dev coordinates: {b}/{variant}")
            values[b][variant] = np.stack([values[b][variant][i] for i in ids])
    create_run(args.output, {"purpose": "Frozen final train-selected bridge analysis" if final else "Held-out-from-search dev analysis, no search feedback or test selection",
        "test_used": bool(final), "final_contract_sha256": file_hash(args.frozen_evaluation) if final else None,
        "bank_sha256": file_hash(args.bank/"manifest.json"),
        "input_hashes": sources, "labels_sha256": file_hash(args.dev/"labels.csv"),
        "script_sha256": file_hash(Path(__file__)), "sample_ids": ids,
        "selection_sha256": file_hash(args.selection), "train_eligible_anchors": eligible,
        "conditioning": "fixed train-selected sets and train-only calibration, source-group bootstrap",
        "semantic_limit": bank["semantic_review"]})
    summaries, reliability, matrices = {}, [], {}
    for name, definition in bank["sets"].items():
        selected = [b["id"] for b in definition]
        for variant in variants:
            cube = np.stack([values[b][variant] for b in selected], axis=2)
            matrix = cube.mean(axis=2)
            key = name+"/"+variant
            summaries[key] = {"multilabel": evaluate_multilabel(gold, matrix, anchors),
                "train_eligible_multilabel": evaluate_multilabel(gold[:, eligible_columns], matrix[:, eligible_columns], eligible),
                "within_sample": evaluate_candidates(dict(zip(ids, gold)), dict(zip(ids, matrix)))}
            matrices[key] = matrix
            np.savez_compressed(args.output/f"{name}-{variant}.npz", values=matrix, cube=cube,
                sample_ids=np.array(ids), anchor_ids=np.array(anchors), bridge_ids=np.array(selected))
            for j, anchor in enumerate(anchors):
                correlations = [spearmanr(cube[:, j, a], cube[:, j, b]).statistic for a, b in combinations(range(3), 2)]
                alpha = cronbach_alpha(cube[:, j, :])
                reliability.append(dict(method=name, variant=variant, anchor_id=anchor,
                    mean_pair_spearman=float(np.mean(correlations)), alpha=alpha["alpha"], alpha_status=alpha["status"]))
        raw_ap = summaries[name+"/raw_logp"]["multilabel"]["macro_ap"]
        ratio_ap = summaries[name+"/reference_log_ratio"]["multilabel"]["macro_ap"]
        if abs(raw_ap-ratio_ap) > 1e-12:
            raise ValueError("Fixed log-ratio shift violated per-anchor AP invariance.")
    groups = np.array([s.group_id or s.id for s in samples])
    blocks = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    rng = np.random.default_rng(42)
    draws = [np.concatenate([blocks[j] for j in rng.integers(len(blocks), size=len(blocks))]) for _ in range(1000)]
    differences = []
    for variant in ["raw_logp", "reference_z"]:
        fixed = matrices["fixed3/"+variant]
        baseline = np.array([evaluate_multilabel(gold[d], fixed[d], anchors)["macro_ap"] for d in draws])
        family = []
        for name in bank["sets"]:
            if name == "fixed3":
                continue
            matrix = matrices[name+"/"+variant]
            boot = np.array([evaluate_multilabel(gold[d], matrix[d], anchors)["macro_ap"] for d in draws]) - baseline
            lo, hi = np.quantile(boot, [.025, .975])
            point = summaries[name+"/"+variant]["multilabel"]["macro_ap"] - summaries["fixed3/"+variant]["multilabel"]["macro_ap"]
            p = (1+np.count_nonzero(np.abs(boot-point) >= abs(point)))/(len(boot)+1)
            family.append(dict(method=name, metric="macro_ap", variant=variant, difference=point,
                ci_lower=lo, ci_upper=hi, centered_bootstrap_p=float(p),
                family=("frozen_primary_z" if variant == "reference_z" else "frozen_secondary_raw") if final else "exploratory_"+variant))
        for row, q in zip(family, bh_fdr([r["centered_bootstrap_p"] for r in family])):
            row["bh_q"] = float(q)
        differences.extend(family)
    write_json(args.output/"metrics.json", summaries)
    pd.DataFrame(reliability).to_csv(args.output/"reliability.csv", index=False)
    pd.DataFrame(differences).to_csv(args.output/"paired-differences.csv", index=False)
    append_event(args.output, "completed")


if __name__ == "__main__":
    main()
