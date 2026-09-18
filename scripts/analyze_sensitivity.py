"""Analyze the frozen bounded E3 instruments; no labels, fitting targets, or selection."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from sensitivity_statistics import geometry_agreement, interval_or_undefined, rank_agreement, template_clusters
from vlanchor import fit_reference, transform
from vlanchor.calibration import _grid
from vlanchor.campaign import append_event, create_run, merge_score_shards
from vlanchor.io import load_samples, load_scores, read_json, save_calibration, write_json
from vlanchor.provenance import file_hash


def read_instrument(roots, ids, spec_hash, accepted=None):
    parts, hashes = [], {}
    for name in roots:
        root = Path(name)
        if (root/"exit_code.txt").read_text().strip() != "0":
            raise ValueError("Incomplete/failed sensitivity run.")
        manifest = read_json(root/"measurement/manifest.json")
        if manifest["spec_sha256"] != spec_hash or manifest["shared_prefill"]:
            raise ValueError("Sensitivity instrument changed or used unadmitted acceleration.")
        if "5000 Ada" not in read_json(root/"measurement/hardware.json")["gpu"]:
            raise ValueError("Sensitivity comparison requires its frozen Ada GPU class.")
        if accepted is not None and (manifest.get("frozen_evaluation") or {}).get("sha256") not in accepted:
            raise ValueError("Protected scoring run lacks an admitted sensitivity contract.")
        for path in sorted((root/"measurement").glob("part-*.parquet")):
            parts.append(load_scores(path))
            hashes[str(path)] = file_hash(path)
            hashes[str(path)+".manifest.json"] = file_hash(path.with_suffix(".parquet.manifest.json"))
        for name in ["manifest.json", "hardware.json", "native-verification.json", "cost.json"]:
            path = root/"measurement"/name
            hashes[str(path)] = file_hash(path)
    return merge_score_shards(parts, ids), hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["bank", "runs", "frozen-evaluation", "output"]:
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--study", choices=["emobank", "dwug"], required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    bank, contract, requests = read_json(args.bank/"manifest.json"), read_json(args.frozen_evaluation), read_json(args.runs)
    if (contract["status"] != "frozen" or contract["study"] != args.study
            or file_hash(args.bank/"manifest.json") != contract["inventory_sha256"]):
        raise ValueError("Sensitivity study or population changed after freeze.")
    for name, digest in contract["analysis_source_sha256"].items():
        if file_hash(root/name) != digest:
            raise ValueError("Sensitivity analysis changed after freeze.")
    for name, digest in bank["files_sha256"].items():
        if file_hash(args.bank/name) != digest:
            raise ValueError("Frozen sensitivity inventory was modified.")
    definition = bank["studies"][args.study]
    if set(requests) != set(definition["variants"]) or set(contract["scoring_contract_sha256"]) != set(requests):
        raise ValueError("Require all predeclared instrument variants.")
    folder = args.bank/args.study
    samples, reference = load_samples(folder/"test128.json"), load_samples(folder/"reference.json")
    ids, ref_ids = [s.id for s in samples], [s.id for s in reference]
    if ids != definition["test_ids"] or ref_ids != definition["reference_ids"] or set(ids) & set(ref_ids):
        raise ValueError("Protected and reference populations differ or overlap.")
    if len(ids) != 128 or any(s.metadata["split"] != "test" for s in samples) or any(s.metadata["split"] != "train" for s in reference):
        raise ValueError("Require128protected materials and the original train-only reference.")
    groups = [s.metadata["sampling_unit"] if args.study == "dwug" else s.group_id for s in samples]
    if any(not group for group in groups):
        raise ValueError("Source units must be identified; no IID fallback.")
    create_run(args.output, {"purpose": "Bounded E3 instrument sensitivity, no external validity claim",
        "final_contract_sha256": file_hash(args.frozen_evaluation), "runs_sha256": file_hash(args.runs),
        "sample_ids": ids, "source_groups": groups, "reference_ids": ref_ids,
        "inference": "1000 paired whole-source draws; descriptive95%intervals, no selection or superiority p-values",
        "pair_inference": "Resample source nodes and induced edges, excluding repeated copies of self edges",
        "clustering": "Reference-z only, fixed k2/seed42/n_init10; predict same original nodes after each source-bootstrap fit",
        "limitations": [bank["alias_scope"], bank["length_scope"], bank["augmentation_scope"]]})
    try:
        instruments, sources, tokens, invariants = {}, {}, [], []
        variants = ["raw_logp", "mean_token_logp", "reference_log_ratio", "reference_z"]
        for name, request in requests.items():
            config_hash = file_hash(folder/f"{name}.json")
            ref, hashes = read_instrument(request["reference_runs"], ref_ids, config_hash)
            if sorted(hashes.values()) != sorted(contract["reference_files_sha256"][name].values()):
                raise ValueError("Train-reference evidence changed after final freeze.")
            sources.update(hashes)
            raw, hashes = read_instrument(request["runs"], ids, config_hash, contract["scoring_contract_sha256"][name])
            sources.update(hashes)
            calibration = fit_reference(ref)
            save_calibration(calibration, args.output/f"{name}-calibration.json")
            scores = transform(raw, calibration)
            matrices = {}
            for variant in variants:
                cube, actual_ids, anchors, bridges, _ = _grid(scores, variant)
                cube = cube[[actual_ids.index(i) for i in ids]]
                matrices[variant] = cube
                np.savez_compressed(args.output/f"{name}-{variant}.npz", cube=cube,
                    sample_ids=np.array(ids), anchor_ids=np.array(anchors), bridge_ids=np.array(bridges))
            instruments[name] = (anchors, matrices)
            a, b = matrices["raw_logp"].mean(axis=2), matrices["reference_log_ratio"].mean(axis=2)
            error = float(np.max(np.abs(cdist(a, a)-cdist(b, b))))
            if not np.allclose(cdist(a, a), cdist(b, b), atol=1e-9, rtol=1e-10):
                raise ValueError("Train reference translation violated Euclidean distance invariance.")
            invariants.append(dict(instrument=name, invariant="raw_logratio_euclidean", max_absolute_error=error))
            for j, anchor in enumerate(anchors):
                for k, bridge in enumerate(bridges):
                    a, b = matrices["raw_logp"][:, j, k], matrices["reference_z"][:, j, k]
                    rho = rank_agreement(a, b)
                    invariants.append(dict(instrument=name, invariant="single_bridge_raw_z_rank",
                        anchor_id=anchor, bridge_id=bridge, spearman=rho if np.isfinite(rho) else None))
                    if np.isfinite(rho) and not np.isclose(rho, 1., atol=1e-12, rtol=0):
                        raise ValueError("Positive affine reference-z changed a single-coordinate rank.")
            for (anchor, bridge), rows in raw.frame.groupby(["anchor_id", "bridge_id"]):
                tokens.append(dict(instrument=name, anchor_id=anchor, bridge_id=bridge,
                    token_min=int(rows.token_count.min()), token_max=int(rows.token_count.max()),
                    token_mean=float(rows.token_count.mean())))
        if args.study == "dwug":
            anchors, matrices = instruments["general256"]
            for count in [64, 128]:
                instruments[f"general{count}-derived"] = (anchors[:count], {v: cube[:, :count] for v, cube in matrices.items()})
        base_anchors, base = instruments["base"]
        for name in ["augmented", "general128-derived"]:
            if name not in instruments:
                continue
            anchors, matrices = instruments[name]
            common = [anchors.index(a) for a in base_anchors]
            invariants.append(dict(instrument=name, invariant="unchanged_anchor_events_under_bank_change",
                max_absolute_error=float(np.max(np.abs(base["raw_logp"]-matrices["raw_logp"][:, common]))),
                scope="Numerical retest, report error without redefining acceptance from observed results"))
            if name == "augmented":
                duplicate = anchors.index("duplicate:"+base_anchors[0])
                invariants.append(dict(instrument=name, invariant="duplicated_anchor_event",
                    max_absolute_error=float(np.max(np.abs(matrices["raw_logp"][:, 0]-matrices["raw_logp"][:, duplicate])))))
        agreements, coordinate_rows, clusters = [], [], {}
        for name, (anchors, matrices) in instruments.items():
            for variant, cube in matrices.items():
                matrix, baseline = cube.mean(axis=2), base[variant].mean(axis=2)
                if name != "base":
                    agreements.append(dict(instrument=name, variant=variant, metric="pair_euclidean_spearman",
                        **geometry_agreement(baseline, matrix, groups)))
                    common = [anchor for anchor in base_anchors if anchor in anchors]
                    for anchor in common:
                        a, b = baseline[:, base_anchors.index(anchor)], matrix[:, anchors.index(anchor)]
                        # Per-coordinate intervals preserve source dependence; constants stay undefined.
                        coordinate_rows.append(dict(instrument=name, variant=variant, anchor_id=anchor,
                            **interval_or_undefined(lambda index, a=a, b=b: rank_agreement(a[index], b[index]), groups)))
            with threadpool_limits(limits=1):
                clusters[name], labels = template_clusters(matrices["reference_z"], groups)
            if labels is not None:
                np.savez_compressed(args.output/f"{name}-clusters.npz", labels=labels, sample_ids=np.array(ids),
                    rows=np.array(["three_bridge_mean", "bridge0", "bridge1", "bridge2"]))
        write_json(args.output/"input-hashes.json", sources)
        write_json(args.output/"invariants.json", invariants)
        write_json(args.output/"clusters.json", clusters)
        pd.DataFrame(tokens).to_csv(args.output/"token-counts.csv", index=False)
        pd.DataFrame(agreements).to_csv(args.output/"geometry-agreement.csv", index=False)
        pd.DataFrame(coordinate_rows).to_csv(args.output/"coordinate-agreement.csv", index=False)
        append_event(args.output, "completed")
    except BaseException as exc:
        append_event(args.output, "failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
