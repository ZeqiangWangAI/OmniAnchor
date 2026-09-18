"""Gate exact per-call vision reuse on frozen32text+8media with both native models and reranker."""
import argparse
import gc
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from omnianchor import measure
from omnianchor.backends.vision_reuse import VisionReuseBackend, reuse_vision_outputs
from omnianchor.campaign import append_event, create_run
from omnianchor.io import load_samples, load_spec, read_json, save_scores, write_json
from omnianchor.official_baselines import QwenRetrieval
from omnianchor.types import Anchor
from verify_native import verify_native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Require Slurm GPU allocation.")
    root = Path(__file__).resolve().parents[1]
    samples = []
    for dataset in ["emobank-reader", "chinese-emobank-sentence"]:
        samples.extend(load_samples(root / "data/prepared/text-smoke-20260910-02" / dataset / "smoke-samples.json"))
    samples.extend(load_samples(root / "data/prepared/multimodal-smoke-20260910-02/samples.json"))
    if len(samples) != 40 or any(s.metadata["split"] not in {"train", "dev"} for s in samples):
        raise ValueError("Require frozen development-only gate inputs.")
    create_run(args.output, {"purpose": "vision-output reuse equivalence; no new neural kernels, KV reuse, or input changes",
        "native_tolerance_nat": .01, "reranker_probability_tolerance": 1e-5,
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "sample_ids": [s.id for s in samples],
        "scope": "clear vision outputs after each score_candidates/paired-query call; exact tensor comparison before reuse"})
    import torch
    torch.manual_seed(42)
    records, verifications = [], {}
    try:
        for model in ["qwen35", "qwen3vl"]:
            spec = load_spec(root / f"configs/smoke/media-{model}.json")
            anchors = (*spec.anchors[:14], Anchor(id="multi_en", surface="a strong emotion"),
                       Anchor(id="multi_zh", surface="自由與關懷", language="zh"))
            spec = spec.model_copy(update={"anchors": anchors, "bridges": spec.bridges[:1]})
            backend = VisionReuseBackend(spec.model, spec.resources, shared_prefill=False)
            backend._ensure_loaded()
            write_json(args.output / model / "identity.json", backend.identity)
            for i, sample in enumerate(samples):
                outputs = {}
                for enabled in [False, True]:
                    backend.reuse_vision_features = enabled
                    torch.cuda.synchronize()
                    started = perf_counter()
                    table = measure([sample], spec, backend=backend)
                    torch.cuda.synchronize()
                    elapsed = perf_counter()-started
                    save_scores(table, args.output / model / str(enabled) / f"part-{i:05d}.parquet")
                    if table.manifest["execution"]["failed_items"]:
                        raise RuntimeError("Native scoring failure; reject reuse.")
                    outputs[enabled] = (table.frame.set_index("anchor_id").loc[[a.id for a in anchors]], elapsed)
                slow, fast = outputs[False][0], outputs[True][0]
                if slow.token_ids.tolist() != fast.token_ids.tolist() or slow.input_fingerprint.tolist() != fast.input_fingerprint.tolist():
                    raise RuntimeError("Native reuse comparison inputs differ.")
                error = float(np.max(np.abs(slow.raw_logp.to_numpy()-fast.raw_logp.to_numpy())))
                token_error = max(abs(a-b) for x, y in zip(slow.token_logps, fast.token_logps) for a, b in zip(x, y))
                record = {"model": model, "sample_id": sample.id, "modality": sample.parts[0].type,
                    "max_abs_sum_nat": error, "max_abs_token_nat": token_error,
                    "reference_seconds": outputs[False][1], "reuse_seconds": outputs[True][1],
                    "vision_calls": backend.last_vision_reuse, "passed": max(error, token_error)<=.01}
                if any(v != 1 for v in backend.last_vision_reuse["computations"].values()):
                    raise RuntimeError("Vision output was recomputed within a reuse scope.")
                records.append(record)
                append_event(args.output, "comparison", **record)
                print(record, flush=True)
            verifications[model] = verify_native(backend, output=args.output / model / "native-verification.json",
                image_sample=next(s for s in samples if s.parts[0].type=="image"),
                video_sample=next(s for s in samples if s.parts[0].type=="video"),
                anchors=[Anchor(id="m1", surface="a strong emotion"), Anchor(id="m2", surface="an emotional response")])
            del backend
            gc.collect()
            torch.cuda.empty_cache()
        models = read_json(root / "configs/models-20260910.json")["models"]
        identifier = "Qwen/Qwen3-VL-Reranker-2B"
        reranker = QwenRetrieval(identifier, models[identifier], root / "research/upstream/qwen3-vl-embedding-393e297", spec.resources)
        for i, sample in enumerate(samples):
            values = {}
            for enabled in [False, True]:
                torch.cuda.synchronize()
                started = perf_counter()
                if enabled:
                    with reuse_vision_outputs(reranker.model) as stats:
                        output = reranker.score(sample, [a.surface for a in anchors], "Retrieve materials related to the concept described by the query.")
                else:
                    output = reranker.score(sample, [a.surface for a in anchors], "Retrieve materials related to the concept described by the query.")
                torch.cuda.synchronize()
                values[enabled] = (output, perf_counter()-started)
            np.savez_compressed(args.output / f"reranker-{i:05d}.npz", original=values[False][0], reused=values[True][0])
            error = float(np.max(np.abs(values[False][0]-values[True][0])))
            record = {"model": "qwen-reranker", "sample_id": sample.id, "modality": sample.parts[0].type,
                "max_abs_probability_error": error, "reference_seconds": values[False][1],
                "reuse_seconds": values[True][1], "vision_calls": stats, "passed": error<=1e-5}
            records.append(record)
            append_event(args.output, "comparison", **record)
            print(record, flush=True)
        passed = all(r["passed"] for r in records) and all(v["status"]=="passed" for v in verifications.values())
        write_json(args.output / "summary.json", {"status": "passed" if passed else "failed", "comparisons": records,
            "models": ["qwen35", "qwen3vl", "qwen-reranker"], "native_verification": {k:v["status"] for k,v in verifications.items()},
            "requires_new_measurement_id": True, "permission_to_mix_old_scores": False})
        append_event(args.output, "completed" if passed else "gate_failed")
        if not passed:
            raise RuntimeError("Vision reuse gate failed; production reuse remains disabled.")
    except BaseException as exc:
        append_event(args.output, "failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
