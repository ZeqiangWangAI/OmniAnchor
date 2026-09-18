"""DWUG development shift and usage-pair validity with explicit source-unit inference."""
import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from dwug_statistics import source_shift
from vlanchor.analysis import bh_fdr
from vlanchor.evaluation import group_bootstrap
from vlanchor.campaign import append_event, create_run
from vlanchor.io import load_samples, read_json, write_json
from vlanchor.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["features", "samples", "labels", "output"]:
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--frozen-evaluation", type=Path)
    args = parser.parse_args()
    samples = load_samples(args.samples)
    final = read_json(args.frozen_evaluation) if args.frozen_evaluation else None
    if final:
        if final["status"] != "frozen" or file_hash(args.samples) != final["samples_sha256"]:
            raise ValueError("Final DWUG population changed.")
        if read_json(args.features/"manifest.json")["final_contract_sha256"] != file_hash(args.frozen_evaluation):
            raise ValueError("Final features were not admitted by the frozen contract.")
        events = (args.features/"events.jsonl").read_text().splitlines()
        if not events or json.loads(events[-1])["status"] != "completed":
            raise ValueError("Final feature assembly did not complete successfully.")
        for name, digest in final["labels_sha256"].items():
            if file_hash(args.labels/name) != digest:
                raise ValueError("Final DWUG labels changed.")
        for name, digest in final["analysis_source_sha256"].items():
            if file_hash(Path(__file__).resolve().parents[1]/name) != digest:
                raise ValueError("Final DWUG statistics changed after freeze.")
    if not samples or any(s.metadata["split"] != ("test" if final else "dev") for s in samples):
        raise ValueError("Unadmitted DWUG evaluation split.")
    ids = [s.id for s in samples]
    targets = np.array([s.target.lemma for s in samples])
    units = np.array([s.metadata["sampling_unit"] for s in samples])
    periods = np.array([s.time for s in samples])
    if any(not u or ".txt" not in u for u in units):
        raise ValueError("Published source document identifiers are required.")
    changes = pd.read_csv(args.labels/"change-labels.csv").set_index("lemma")
    if changes.index.duplicated().any() or set(changes.index) != set(targets):
        raise ValueError("Change gold does not match the complete target population.")
    judgments = pd.read_csv(args.labels/"judgments.csv")
    if not judgments.judgment.isin([0, 1, 2, 3, 4]).all():
        raise ValueError("Missing or out-of-scale published judgment; no silent removal.")
    excluded = judgments[judgments.judgment == 0]
    judgments = judgments[judgments.judgment > 0].copy()
    if not judgments.judgment.between(1, 4).all():
        raise ValueError("Unexpected usage-relatedness scale.")
    pairs = np.sort(judgments[["identifier1", "identifier2"]].to_numpy(str), axis=1)
    judgments["left"], judgments["right"] = pairs[:, 0], pairs[:, 1]
    pairs = judgments.groupby(["lemma", "left", "right"], sort=True).judgment.mean().reset_index()
    lookup = {s: i for i, s in enumerate(ids)}
    left = np.array([lookup[i] for i in pairs.left])
    right = np.array([lookup[i] for i in pairs.right])
    if not np.array_equal(targets[left], pairs.lemma.to_numpy()) or not np.array_equal(targets[right], pairs.lemma.to_numpy()):
        raise ValueError("Usage pair has mismatched target endpoints.")
    features = {}
    feature_info = read_json(args.features/"features.json")
    for name in feature_info:
        with np.load(args.features/f"{name}.npz", allow_pickle=False) as data:
            row_ids = data["sample_ids"].tolist()
            if len(row_ids) != len(set(row_ids)) or not set(ids) <= set(row_ids):
                raise ValueError("Missing/duplicate dev features.")
            features[name] = data["values"][[row_ids.index(i) for i in ids]].astype(float)
    if not all(np.isfinite(v).all() for v in features.values()):
        raise ValueError("Nonfinite geometry.")
    unit_targets = {}
    for u, target in zip(units, targets):
        unit_targets.setdefault(u, set()).add(target)
    create_run(args.output, {"purpose": "DWUG frozen final evaluation" if final else "DWUG dev diagnostic; all fixed feature geometries, no target selection",
        "test_used": bool(final), "final_contract_sha256": file_hash(args.frozen_evaluation) if final else None,
        "sample_ids": ids, "targets": sorted(set(targets)), "seed": 42,
        "script_sha256": file_hash(Path(__file__)), "statistics_sha256": file_hash(Path(__file__).with_name("dwug_statistics.py")),
        "inputs": {str(p): file_hash(p) for p in [args.samples, args.labels/"change-labels.csv", args.labels/"judgments.csv"]},
        "features_sha256": {name: file_hash(args.features/f"{name}.npz") for name in features},
        "source_documents_crossing_targets": {u: sorted(t) for u, t in unit_targets.items() if len(t) > 1},
        "within_target_ci": "1000 source-document bootstrap draws, fixed row-weighted period distributions",
        "permutation": "1000 whole-document period reassignments within filename genre; conditional exchangeability assumption, not causal identification",
        "sensitivity": "100 row-count-balanced draws,100within-periodsourcehalfsplits per period; descriptive controls, not confidence intervals",
        "across_target_ci": "1000 target-resampled draws conditional on this corpus; shared documents across targets limit independence",
        "pair_aggregation": "exclude undecidable0; mean all remaining1-4ratings per unordered within-target pair",
        "network": "full cosine adjacency for all methods; native-raw GraphML; unweighted k=2KMeans only descriptive, no human-cluster accuracy claim"})
    shifts, pair_rows, clusters = [], [], []
    pair_predictions = {}
    with threadpool_limits(limits=1):
        for name, values in features.items():
            norms = np.linalg.norm(values, axis=1)
            if np.any(norms == 0):
                raise ValueError("Zero-vector cosine undefined; no zero imputation.")
            normalized = values/norms[:, None]
            predictions = np.einsum("ij,ij->i", normalized[left], normalized[right])
            pair_predictions[name] = predictions
            for target in sorted(set(targets)):
                index = np.flatnonzero(targets == target)
                stats = source_shift(values[index], periods[index], units[index])
                shifts.append({"method": name, "target": target, "human_change": float(changes.loc[target, "change_graded"]), **stats})
                mask = pairs.lemma.to_numpy() == target
                pair_rows.append({"method": name, "target": target, "n_pairs": int(mask.sum()),
                    "spearman": spearmanr(pairs.judgment.to_numpy()[mask], predictions[mask]).statistic})
                folder = args.output/name
                folder.mkdir(exist_ok=True)
                adjacency = np.clip(normalized[index] @ normalized[index].T, -1, 1)
                np.savez_compressed(folder/f"{target}-adjacency.npz", adjacency=adjacency, sample_ids=np.array(ids)[index])
                if name == "native-raw_logp":
                    graph = nx.Graph()
                    graph.add_nodes_from((ids[i], {"period": str(periods[i]), "sampling_unit": str(units[i])}) for i in index)
                    graph.add_weighted_edges_from((ids[index[a]], ids[index[b]], float(adjacency[a, b])) for a in range(len(index)) for b in range(a+1, len(index)))
                    nx.write_graphml(graph, folder/f"{target}.graphml")
                if len(np.unique(values[index], axis=0)) < 2:
                    raise ValueError("Fewer than2distinct sample vectors; fixedk2undefined.")
                assignment = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(values[index])
                clusters.extend(dict(method=name, target=target, sample_id=ids[i], cluster=int(c), k=2, seed=42) for i, c in zip(index, assignment))
                print(name, target, "complete", flush=True)
    shift_table = pd.DataFrame(shifts)
    for name in features:
        mask = shift_table.method == name
        shift_table.loc[mask, "within_method_target_bh_q"] = bh_fdr(shift_table.loc[mask, "genre_stratified_permutation_p"].to_numpy())
    rng = np.random.default_rng(42)
    target_order = sorted(set(targets))
    draws = [rng.integers(len(target_order), size=len(target_order)) for _ in range(1000)]
    correlations, rho_boots = [], {}
    for name in features:
        frame = shift_table[shift_table.method == name].set_index("target").loc[target_order]
        for metric in ["centroid_distance", "energy_distance", "balanced_centroid_mean", "balanced_energy_mean"]:
            x, y = frame[metric].to_numpy(), frame.human_change.to_numpy()
            boot = np.array([spearmanr(x[d], y[d]).statistic for d in draws])
            valid = np.isfinite(boot)
            rho_boots[(name, metric)] = (float(spearmanr(x, y).statistic), boot)
            interval = np.quantile(boot[valid], [.025, .975]) if valid.sum() >= 800 else [np.nan, np.nan]
            correlations.append(dict(method=name, metric=metric, spearman=spearmanr(x, y).statistic,
                ci_lower=interval[0], ci_upper=interval[1], valid_bootstraps=int(valid.sum()), n_targets=len(target_order)))
    differences = []
    for metric in ["energy_distance", "centroid_distance"]:
        point, boot = rho_boots[("native-raw_logp", metric)]
        for comparator in ["e5-original", "qwen-embedding-original", "qwen-reranker-anchor"]:
            other, other_boot = rho_boots[(comparator, metric)]
            delta = boot-other_boot
            delta = delta[np.isfinite(delta)]
            interval = np.quantile(delta, [.025, .975]) if len(delta) >= 800 else [np.nan, np.nan]
            pv = (1+(np.abs(delta-(point-other)) >= abs(point-other)).sum())/(len(delta)+1) if len(delta) >= 800 else np.nan
            differences.append(dict(metric=metric, comparator=comparator, difference=point-other,
                ci_lower=interval[0], ci_upper=interval[1], centered_bootstrap_p=pv, valid_bootstraps=len(delta)))
    difference_table = pd.DataFrame(differences)
    mask = difference_table.metric == "energy_distance"
    difference_table.loc[mask, "frozen_energy_family_bh_q" if final else "exploratory_energy_family_bh_q"] = bh_fdr(difference_table.loc[mask, "centered_bootstrap_p"].to_numpy())
    difference_table.to_csv(args.output/"paired-change-differences.csv", index=False)
    raw_shift = shift_table[shift_table.method == "native-raw_logp"].set_index("target")
    ratio_shift = shift_table[shift_table.method == "native-reference_log_ratio"].set_index("target").loc[raw_shift.index]
    np.testing.assert_allclose(raw_shift[["centroid_distance", "energy_distance"]],
        ratio_shift[["centroid_distance", "energy_distance"]], rtol=1e-10, atol=1e-8)
    shift_table.to_csv(args.output/"per-target-shift.csv", index=False)
    pd.DataFrame(correlations).to_csv(args.output/"change-validity.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(args.output/"per-target-relatedness.csv", index=False)
    pd.DataFrame(clusters).to_csv(args.output/"descriptive-clusters.csv", index=False)
    excluded.to_csv(args.output/"excluded-undecidable-judgments.csv", index=False)
    pairs.to_csv(args.output/"aggregated-pair-gold.csv", index=False)
    np.savez_compressed(args.output/"pair-predictions.npz", **pair_predictions)
    pair_validity = {}
    for name, v in pair_predictions.items():
        estimate = group_bootstrap(lambda index: spearmanr(pairs.judgment.to_numpy()[index], v[index]).statistic,
            pairs.lemma.to_numpy())
        pair_validity[name] = {"spearman": float(spearmanr(pairs.judgment, v).statistic),
            "n_pairs": len(pairs), "target_group_bootstrap": estimate,
            "inference": "Whole target clusters retain shared endpoints/annotators within target; conditional on observed corpus/ratings, not IID pair inference. Cross-target shared documents and annotators limit independence; few retained targets limit interval precision."}
    write_json(args.output/"pooled-pair-validity.json", pair_validity)
    append_event(args.output, "completed")


if __name__ == "__main__":
    main()
