"""Assemble frozen train/dev score shards and official baseline matrices with exact ID joins."""
import argparse
from pathlib import Path

import numpy as np

from omnianchor import fit_reference, to_matrix, transform
from omnianchor.campaign import append_event, combine_bridge_tables, create_run, merge_score_shards
from omnianchor.io import load_samples, load_scores, read_json, save_calibration, write_json
from omnianchor.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--native-runs", nargs="*", type=Path, default=[])
    parser.add_argument("--baseline-runs", nargs="+", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=["e5", "qwen-embedding", "qwen-reranker"],
                        default=["e5", "qwen-embedding", "qwen-reranker"])
    parser.add_argument("--reference-candidates", type=Path)
    parser.add_argument("--reference-run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-evaluation", type=Path)
    args = parser.parse_args()
    if len(set(args.methods)) != len(args.methods):
        raise ValueError("Duplicate baseline methods.")
    shard_manifest = read_json(args.shards / "manifest.json")
    samples = [s for f in shard_manifest["files"] for s in load_samples(args.shards / f["file"])]
    ids = [s.id for s in samples]
    final = read_json(args.frozen_evaluation) if args.frozen_evaluation else None
    if final:
        if final["status"] != "frozen" or ids != final["sample_ids"]:
            raise ValueError("Final feature population differs from frozen protocol.")
        if file_hash(args.shards/"manifest.json") != final["shards_manifest_sha256"]:
            raise ValueError("Final shard inventory changed.")
        if args.methods != final["baseline_methods"] or not args.reference_run:
            raise ValueError("Final features require every frozen method and original reference run.")
        for name, digest in final["reference_score_sha256"].items():
            if file_hash(args.reference_run/name) != digest:
                raise ValueError("Frozen train reference changed.")
        if file_hash(Path(__file__)) != final["feature_assembly_sha256"]:
            raise ValueError("Final feature assembler changed after freeze.")
    if any(s.metadata["split"] not in ({"test"} if final else {"train", "dev"}) for s in samples):
        raise ValueError("Unadmitted split in feature assembly.")
    for root in args.native_runs + args.baseline_runs + ([args.reference_run] if args.reference_run else []):
        if (root / "exit_code.txt").read_text().strip() != "0":
            raise ValueError(f"Incomplete/failed run {root}; preserve and repair explicitly.")
    if final:
        accepted = set(final["scoring_contract_sha256"].values())
        for root in args.native_runs + args.baseline_runs:
            folders = [root/"measurement"] if root in args.native_runs else [root/m for m in args.methods]
            for folder in folders:
                if read_json(folder/"manifest.json")["frozen_evaluation"]["sha256"] not in accepted:
                    raise ValueError("A scoring run is outside the frozen shard contracts.")
    create_run(args.output, {"purpose": "frozen final features, labels not read" if final else "development features, labels not read", "sample_ids": ids,
        "final_contract_sha256": file_hash(args.frozen_evaluation) if final else None,
        "shard_manifest_sha256": file_hash(args.shards / "manifest.json"),
        "native_runs": [str(p) for p in args.native_runs], "baseline_runs": [str(p) for p in args.baseline_runs],
        "baseline_methods": args.methods})
    try:
        parts = [p for root in args.native_runs for p in sorted((root / "measurement").glob("part-*.parquet"))]
        bridges = []
        reference_paths = []
        anchors = read_json(args.baseline_runs[0] / args.methods[0] / "manifest.json")["anchor_ids"]
        if args.native_runs:
            if bool(args.reference_candidates) == bool(args.reference_run):
                raise ValueError("Native features require exactly one frozen reference source.")
            raw = load_scores(parts[0])
            if [a["id"] for a in raw.manifest["anchors"]] != anchors:
                raise ValueError("Native and baseline coordinates differ.")
            bridges = [b["id"] for b in raw.manifest["bridges"]]
            if args.reference_candidates:
                reference_paths = [args.reference_candidates / b / "reference.parquet" for b in bridges]
                reference = combine_bridge_tables([load_scores(p) for p in reference_paths])
            else:
                reference_paths = sorted((args.reference_run / "measurement").glob("part-*.parquet"))
                references = [load_scores(p) for p in reference_paths]
                reference_ids = [s["id"] for t in references for s in t.manifest["samples"]]
                reference = merge_score_shards(references, reference_ids)
            if any(s["metadata"]["split"] != "train" for s in reference.manifest["samples"]):
                raise ValueError("Reference is not train-only.")
            if {s["id"] for s in reference.manifest["samples"]} & set(ids):
                raise ValueError("Reference overlaps probe train/dev.")
            if {s["group_id"] for s in reference.manifest["samples"]} & {s.group_id for s in samples}:
                raise ValueError("Reference source groups overlap probe train/dev.")
            calibration = fit_reference(reference)
            save_calibration(calibration, args.output / "calibration.json")
            native_rows = {variant: {} for variant in ["raw_logp", "reference_log_ratio", "reference_z"]}
            for path in parts:
                part = load_scores(path)
                part_ids = [s["id"] for s in part.manifest["samples"]]
                part = merge_score_shards([part], part_ids)
                scores = transform(part, calibration)
                for variant, rows in native_rows.items():
                    matrix = to_matrix(scores, variant=variant, missing="error")
                    if set(matrix.anchor_ids) != set(anchors):
                        raise ValueError("Native part coordinates differ.")
                    columns = [matrix.anchor_ids.index(a) for a in anchors]
                    for i, identifier in enumerate(matrix.sample_ids):
                        if identifier in rows:
                            raise ValueError("Duplicate native sample across parts.")
                        rows[identifier] = matrix.values[i, columns].copy()
            if any(set(rows) != set(ids) for rows in native_rows.values()):
                raise ValueError("Native shard coverage differs; no implicit filtering.")
        sources = {}

        def save(name, values, columns, direct):
            if values.shape != (len(ids), len(columns)) or not np.isfinite(values).all():
                raise ValueError(f"Invalid matrix {name}")
            np.savez_compressed(args.output / f"{name}.npz", values=values, sample_ids=np.array(ids),
                                column_ids=np.array(columns))
            sources[name] = {"direct_anchor_scores": direct, "columns": len(columns)}

        for variant in (["raw_logp", "reference_log_ratio", "reference_z"] if args.native_runs else []):
            ordered = np.stack([native_rows[variant][i] for i in ids])
            save("native-"+variant, ordered, anchors, True)
        for method in args.methods:
            rows, vectors = {}, {}
            identity = None
            for root in args.baseline_runs:
                manifest = read_json(root / method / "manifest.json")
                current = {k: manifest.get(k) for k in ["method", "model_id", "revision", "spec_sha256", "instruction", "precision", "preprocessing", "vision_reuse"]}
                if identity is not None and current != identity:
                    raise ValueError("Baseline shard identities differ.")
                identity = current
                with np.load(root / method / "matrix.npz", allow_pickle=False) as data:
                    columns = data["anchor_ids"].tolist()
                    score_values, feature_values = data["scores"], data["features"]
                    if set(columns) != set(anchors):
                        raise ValueError("Baseline anchor IDs differ.")
                    for i, identifier in enumerate(data["sample_ids"].tolist()):
                        if identifier in rows:
                            raise ValueError("Duplicate baseline sample ID.")
                        rows[identifier] = score_values[i, [columns.index(a) for a in anchors]]
                        if feature_values.size:
                            vectors[identifier] = feature_values[i].copy()
            if set(rows) != set(ids):
                raise ValueError("Baseline coverage differs; no implicit intersection or dropping.")
            save(method+"-anchor", np.stack([rows[i] for i in ids]), anchors, True)
            if vectors:
                features = np.stack([vectors[i] for i in ids])
                save(method+"-original", features, [f"feature_{j:04d}" for j in range(features.shape[1])], False)
            write_json(args.output / f"{method}-identity.json", identity)
        write_json(args.output / "features.json", sources)
        write_json(args.output / "input-hashes.json", {str(p): file_hash(p) for p in parts} | {
            str(root / method / name): file_hash(root / method / name) for root in args.baseline_runs
            for method in args.methods for name in ["matrix.npz", "manifest.json"]} | {
            str(p): file_hash(p) for p in reference_paths})
        append_event(args.output, "completed")
    except BaseException as exc:
        append_event(args.output, "failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
