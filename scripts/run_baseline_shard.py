"""Evaluate official retrieval baselines on frozen development inputs, without labels."""
import argparse
from contextlib import nullcontext
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from vlanchor.campaign import append_event, create_run
from vlanchor.io import load_samples, load_spec, read_json, write_json
from vlanchor.official_baselines import E5Embedding, QwenRetrieval
from vlanchor.provenance import file_hash, runtime_manifest
from vlanchor.types import Part, Sample
from vlanchor.backends.vision_reuse import reuse_vision_outputs
from vision_reuse_admission import admit_vision_reuse
from frozen_evaluation import validate_frozen_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["e5", "qwen-embedding", "qwen-reranker"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vision-reuse-gate", type=Path)
    parser.add_argument("--frozen-evaluation", type=Path)
    args = parser.parse_args()
    spec, samples = load_spec(args.config), load_samples(args.samples)
    final = (validate_frozen_evaluation(args.frozen_evaluation, args.samples, args.config, args.method)
             if args.frozen_evaluation else None)
    if not os.environ.get("SLURM_JOB_ID") or (not final and any(s.metadata.get("split") not in {"train", "dev"} for s in samples)):
        raise ValueError("Require Slurm and development-only samples.")
    root = Path(__file__).resolve().parents[1]
    models = read_json(root / "configs/models-20260910.json")["models"]
    ids = {"e5": "intfloat/multilingual-e5-large", "qwen-embedding": "Qwen/Qwen3-VL-Embedding-2B",
           "qwen-reranker": "Qwen/Qwen3-VL-Reranker-2B"}
    model_id = ids[args.method]
    # Fix task instructions before scores or evaluation labels are available.
    instruction = {
        "evokes_emotion": "Retrieve materials that evoke the emotion described by the query in a reader.",
        "expresses_value": "Retrieve materials that express the human value described by the query.",
        "associated_with": "Retrieve materials related to the concept described by the query.",
        "means_in_context": "Retrieve usages where the word marked with <target> tags has the meaning described by the query.",
    }[spec.bridges[0].relation]
    manifest = {"method": args.method, "model_id": model_id, "revision": models[model_id],
                "slurm_job_id": os.environ["SLURM_JOB_ID"], "runtime": runtime_manifest(),
                "samples_sha256": file_hash(args.samples), "spec_sha256": file_hash(args.config),
                "sample_ids": [s.id for s in samples], "anchor_ids": [a.id for a in spec.anchors],
                "instruction": (instruction if args.method == "qwen-reranker" else
                                "query: " if args.method == "e5" else "Represent the user's input."),
                "e5_roles": {"material": "query", "anchor": "query"} if args.method == "e5" else None,
                "precision": "bf16; FP32 pooling/normalization", "batch_size": 1,
                "preprocessing": "shared VLanchor strict native media budget; no truncation or NULL fallback",
                "purpose": "development_only_no_final_test", "source_hashes": {
                    str(p.relative_to(root)): file_hash(p) for folder in ["src", "scripts", "research/upstream"]
                    for p in (root / folder).rglob("*.py")}}
    manifest["vision_reuse"] = admit_vision_reuse(args.vision_reuse_gate) if args.vision_reuse_gate else None
    manifest["frozen_evaluation"] = final
    if final:
        manifest["purpose"] = "frozen_final_evaluation_no_method_selection"
    create_run(args.output, manifest)
    start = perf_counter()
    try:
        import torch
        torch.manual_seed(42)
        surfaces = [a.surface for a in spec.anchors]
        if args.method == "e5":
            if any(p.type != "text" for s in samples for p in s.parts):
                raise ValueError("E5 supports text only; no dropped media.")
            model = E5Embedding(models[model_id])
            # Symmetric feature geometry uses query on both sides, as recommended for non-retrieval E5 tasks.
            anchor_features = model.encode(surfaces, role="query")
        else:
            model = QwenRetrieval(model_id, models[model_id], root / "research/upstream/qwen3-vl-embedding-393e297", spec.resources)
            if args.method == "qwen-embedding":
                anchor_features = model.encode([Sample(id=a.id, parts=(Part(type="text", text=a.surface),)) for a in spec.anchors])
        torch.cuda.reset_peak_memory_stats()
        measured = perf_counter()
        features, scores = [], []
        for i, sample in enumerate(samples):
            if args.method == "e5":
                vector = model.encode(["\n".join(p.text for p in sample.parts)], role="query")[0]
                features.append(vector)
                values = vector @ anchor_features.T
            elif args.method == "qwen-embedding":
                vector = model.encode([sample])[0]
                features.append(vector)
                values = vector @ anchor_features.T
                write_json(args.output / f"input-{i:05d}.json", model.last_input_evidence)
            else:
                with reuse_vision_outputs(model.model) if args.vision_reuse_gate else nullcontext():
                    values = model.score(sample, surfaces, instruction)
            if not np.isfinite(values).all():
                raise ValueError("Baseline produced nonfinite values.")
            scores.append(values)
            np.savez_compressed(args.output / f"part-{i:05d}.npz", scores=values,
                                sample_id=sample.id, anchor_ids=np.array([a.id for a in spec.anchors]),
                                features=features[-1] if features else np.empty(0))
            append_event(args.output, "sample_completed", sample_id=sample.id)
            print(f"{i+1}/{len(samples)} {sample.id}", flush=True)
        torch.cuda.synchronize()
        np.savez_compressed(args.output / "matrix.npz", scores=np.stack(scores),
                            features=np.stack(features) if features else np.empty((0, 0)),
                            sample_ids=np.array([s.id for s in samples]),
                            anchor_ids=np.array([a.id for a in spec.anchors]))
        cost = {"measurement_seconds": perf_counter() - measured, "total_seconds": perf_counter() - start,
                "samples": len(samples), "anchors": len(spec.anchors),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        write_json(args.output / "cost.json", cost)
        append_event(args.output, "completed", cost=cost)
    except BaseException as exc:
        append_event(args.output, "failed", error=repr(exc), elapsed_seconds=perf_counter() - start)
        raise


if __name__ == "__main__":
    main()
