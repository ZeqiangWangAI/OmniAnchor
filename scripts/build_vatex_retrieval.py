"""Convert complete frozen VATEX feature matrices to caption–video cosine scores."""
import argparse
import json
from pathlib import Path

import numpy as np

from omnianchor.campaign import append_event, create_run
from omnianchor.io import load_samples, read_json, write_json
from omnianchor.provenance import file_hash


def cosine_scores(values, sample_ids, query_ids, candidate_ids):
    values = np.asarray(values, dtype=float)
    if (values.ndim != 2 or values.shape[0] != len(sample_ids) or not values.shape[1]
            or len(set(sample_ids)) != len(sample_ids) or not np.isfinite(values).all()):
        raise ValueError("Require a complete finite feature matrix with unique row IDs.")
    if (len(set(query_ids+candidate_ids)) != len(query_ids)+len(candidate_ids)
            or not set(query_ids+candidate_ids) <= set(sample_ids)):
        raise ValueError("Missing, duplicate, or overlapping query/gallery IDs.")
    index = {name: i for i, name in enumerate(sample_ids)}
    queries = values[[index[name] for name in query_ids]]
    candidates = values[[index[name] for name in candidate_ids]]
    qnorm, cnorm = np.linalg.norm(queries, axis=1), np.linalg.norm(candidates, axis=1)
    if np.any(qnorm == 0) or np.any(cnorm == 0):
        raise ValueError("Zero feature vectors have undefined cosine; no silent exclusion.")
    return (queries/qnorm[:, None]) @ (candidates/cnorm[:, None]).T


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["features", "videos", "captions", "output"]:
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--frozen-evaluation", type=Path)
    args = parser.parse_args()
    videos, captions = load_samples(args.videos), load_samples(args.captions)
    final = read_json(args.frozen_evaluation) if args.frozen_evaluation else None
    samples = videos+captions
    if not videos or not captions or any(s.metadata.get("split") != ("test" if final else "dev") for s in samples):
        raise ValueError("Require the complete admitted evaluation split.")
    if any(len(s.parts) != 1 or s.parts[0].type != "video" for s in videos):
        raise ValueError("Video gallery must not contain caption text.")
    if any(len(s.parts) != 1 or s.parts[0].type != "text" for s in captions) or len({s.language for s in captions}) != 1:
        raise ValueError("Caption queries must be text only and one language.")
    vi, ci = sorted(s.id for s in videos), sorted(s.id for s in captions)
    if set(s.pair_id for s in captions) != set(vi):
        raise ValueError("Published caption pairs must cover the entire gallery.")
    if any(sum(s.pair_id == identifier for s in captions) != 10 for identifier in vi):
        raise ValueError("Require all ten published captions per video.")
    events = (args.features/"events.jsonl").read_text().splitlines()
    if not events or json.loads(events[-1])["status"] != "completed":
        raise ValueError("Feature assembly did not complete.")
    feature_manifest = read_json(args.features/"manifest.json")
    methods = read_json(args.features/"features.json")
    if final:
        if final["status"] != "frozen" or final["builder_sha256"] != file_hash(Path(__file__)):
            raise ValueError("Retrieval builder changed after freeze.")
        if (final["videos_sha256"] != file_hash(args.videos) or final["captions_sha256"] != file_hash(args.captions)
                or final["methods"] != sorted(methods)
                or final["feature_contract_sha256"] != feature_manifest.get("final_contract_sha256")):
            raise ValueError("Final population, methods, or feature admission changed.")
    elif feature_manifest.get("final_contract_sha256"):
        raise ValueError("Protected features require their final retrieval admission.")
    create_run(args.output, {"purpose": "VATEX cosine retrieval matrices; no labels fitted or parameters selected",
        "test_used": bool(final), "final_contract_sha256": file_hash(args.frozen_evaluation) if final else None,
        "features_manifest_sha256": file_hash(args.features/"manifest.json"),
        "videos_sha256": file_hash(args.videos), "captions_sha256": file_hash(args.captions),
        "builder_sha256": file_hash(Path(__file__)), "query_ids": ci, "candidate_ids": vi,
        "methods": sorted(methods), "geometry": "cosine of saved features; no centering, projection, or fitted retrieval model",
        "native_calibration": "Existing joint feature assembly uses the same fixed train reference for both modalities",
        "reranker_anchor_scope": "Cosine in anchor yes-probability coordinates; distinct from direct caption-video reranking"})
    try:
        hashes = {}
        for name, definition in sorted(methods.items()):
            path = args.features/f"{name}.npz"
            with np.load(path, allow_pickle=False) as data:
                ids, columns = data["sample_ids"].tolist(), data["column_ids"].tolist()
                if ids != feature_manifest["sample_ids"] or len(columns) != definition["columns"] or len(set(columns)) != len(columns):
                    raise ValueError("Feature row or coordinate identities differ from complete assembly.")
                scores = cosine_scores(data["values"], ids, ci, vi)
            np.savez_compressed(args.output/f"{name}.npz", scores=scores,
                query_ids=np.array(ci), candidate_ids=np.array(vi))
            hashes[name] = file_hash(path)
        write_json(args.output/"input-hashes.json", hashes)
        write_json(args.output/"coverage.json", {"videos": len(vi), "captions": len(ci),
            "methods": len(methods), "excluded_queries": [], "excluded_candidates": []})
        append_event(args.output, "completed")
    except BaseException as exc:
        append_event(args.output, "failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
