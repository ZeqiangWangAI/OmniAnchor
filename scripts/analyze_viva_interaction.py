"""Paired output-score modality interaction; not an internal fusion or validity test."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_viva import aligned_scores
from vlanchor.campaign import append_event, create_run
from vlanchor.evaluation import group_bootstrap
from vlanchor.io import load_samples, read_json, write_json
from vlanchor.provenance import file_hash


def interaction_scores(image_text, image, text, empty):
    arrays = [np.asarray(x, dtype=float) for x in [image_text, image, text, empty]]
    if any(a.shape != arrays[0].shape or not np.isfinite(a).all() for a in arrays):
        raise ValueError("Interaction requires aligned, complete finite scores.")
    return arrays[0]-arrays[1]-arrays[2]+arrays[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["base-runs", "control-runs"]:
        parser.add_argument("--"+name, type=Path, nargs="+", required=True)
    for name in ["samples", "candidates", "output"]:
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--base-contract", type=Path)
    parser.add_argument("--control-contract", type=Path)
    args = parser.parse_args()
    samples = load_samples(args.samples)
    ids = [s.id for s in samples]
    final = bool(args.base_contract and args.control_contract)
    if bool(args.base_contract) != bool(args.control_contract):
        raise ValueError("Both protected scoring contracts are required.")
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError("Require unique recipients.")
    if any(s.metadata["split"] not in ({"test"} if final else {"train", "dev"}) for s in samples):
        raise ValueError("Unadmitted protected or unspecified samples.")
    if final:
        for path in [args.base_contract, args.control_contract]:
            c = read_json(path)
            if c["status"] != "frozen" or c["samples_sha256"] != file_hash(args.samples) or c["sample_ids"] != ids:
                raise ValueError("Protected interaction population differs from scoring admission.")
            if c["viva_input_sha256"]["candidates.json"] != file_hash(args.candidates):
                raise ValueError("Candidate identity changed.")
        control = read_json(args.control_contract)
        if control["interaction_analysis_sha256"] != file_hash(Path(__file__)):
            raise ValueError("Interaction analysis changed after freeze.")
    candidate_bank = read_json(args.candidates)
    expected = {i: [a["id"] for a in candidate_bank[i]] for i in ids}
    scores, identities, hardware, hashes = {}, {}, {}, {}
    methods = ["native", "qwen-embedding", "qwen-reranker"]
    for runs, conditions, contract in [
        (args.base_runs, ["image_action", "action_only"], args.base_contract),
        (args.control_runs, ["image_only", "empty"], args.control_contract),
    ]:
        for run in runs:
            if (run/"exit_code.txt").read_text().strip() != "0":
                raise ValueError("Incomplete scoring run; no filtering failed conditions.")
            for method in methods:
                folder = run/method
                if not folder.exists():
                    continue
                m = read_json(folder/"manifest.json")
                if final and m["frozen_evaluation"]["sha256"] != file_hash(contract):
                    raise ValueError("Scoring used a different final contract.")
                if not set(conditions) <= set(m["conditions"]) or m["method"] != method:
                    raise ValueError("Missing or mislabeled scoring conditions.")
                shard_ids = m["sample_ids"]
                if len(shard_ids) != len(set(shard_ids)) or not set(shard_ids) <= set(ids):
                    raise ValueError("Unexpected recipient coverage.")
                identity = {k: m[k] for k in ["spec", "vision_reuse"]}
                identity["model"] = read_json(folder/"model-identity.json")
                gpu = read_json(folder/"cost.json")["gpu"]
                if method in identities and (identity != identities[method] or gpu != hardware[method]):
                    raise ValueError("Interaction conditions have different instruments or GPU classes.")
                identities[method], hardware[method] = identity, gpu
                for condition in conditions:
                    values, inputs = aligned_scores(folder/condition, method == "native", {i: expected[i] for i in shard_ids})
                    hashes.update(inputs)
                    destination = scores.setdefault((method, condition), {})
                    if set(destination) & set(values):
                        raise ValueError("Duplicate condition scores; cannot select among repeated runs.")
                    destination.update(values)
    if len(scores) != 12 or any(set(v) != set(ids) for v in scores.values()):
        raise ValueError("Require all three methods and four complete paired conditions.")
    groups = [s.group_id for s in samples]
    if any(not g for g in groups):
        raise ValueError("Recipient image groups are required.")
    create_run(args.output, {"purpose": "Output-score modality interaction only", "test_used": final,
        "sample_ids": ids, "source_sha256": hashes, "hardware": hardware,
        "input_sha256": {str(p): file_hash(p) for p in [args.samples, args.candidates]},
        "definition": "delta=s(image,action)-s(image,empty)-s(empty,action)+s(empty,empty)",
        "native_units": "mean-bridge log probability in nat", "embedding_units": "anchor cosine similarity",
        "reranker_units": "official yes probability; magnitudes not comparable across methods",
        "claim_boundary": "Nonadditivity of output scores; no identification of internal fusion or psychological validity. Visual utility is evaluated separately by paired AP controls.",
        "intervals": "1000 recipient-image-group bootstrap draws; descriptive 95% intervals, no p-values for nonnegative RMS"})
    try:
        rows, summary = [], []
        for method in methods:
            means, rms = [], []
            for i in ids:
                delta = interaction_scores(scores[method, "image_action"][i], scores[method, "image_only"][i],
                    scores[method, "action_only"][i], scores[method, "empty"][i])
                means.append(delta.mean())
                rms.append(np.sqrt(np.mean(delta**2)))
                rows.extend(dict(method=method, sample_id=i, anchor_id=a, delta=float(d)) for a, d in zip(expected[i], delta))
            for metric, values in [("mean_signed_delta", np.array(means)), ("mean_recipient_rms_delta", np.array(rms))]:
                result = group_bootstrap(lambda index: values[index].mean(), groups)
                summary.append(dict(method=method, metric=metric, **result))
        pd.DataFrame(rows).to_csv(args.output/"per-coordinate-interactions.csv", index=False)
        pd.DataFrame(summary).to_csv(args.output/"interaction-summary.csv", index=False)
        write_json(args.output/"coverage.json", {"recipients": len(ids), "source_groups": len(set(groups)), "excluded": []})
        append_event(args.output, "completed")
    except BaseException as exc:
        append_event(args.output, "failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
