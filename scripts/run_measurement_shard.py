"""Run an immutable, resumable-by-new-run measurement shard on a Slurm CUDA allocation."""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from time import perf_counter

from vlanchor import measure
from vlanchor.backends.vision_reuse import VisionReuseBackend
from vlanchor.campaign import append_event, create_run
from vlanchor.io import load_samples, load_spec, save_scores, write_json
from vlanchor.provenance import file_hash, runtime_manifest
from vlanchor.types import Anchor
from verify_native import verify_native
from vision_reuse_admission import admit_vision_reuse
from frozen_evaluation import validate_frozen_evaluation


class TimedBackend(VisionReuseBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward_seconds = 0.0
        self.encoding_seconds = 0.0

    def _reference(self, item):
        import torch
        torch.cuda.synchronize()
        start = perf_counter()
        try:
            return super()._reference(item)
        finally:
            torch.cuda.synchronize()
            self.forward_seconds += perf_counter() - start

    def _encode(self, *args, **kwargs):
        start = perf_counter()
        try:
            return super()._encode(*args, **kwargs)
        finally:
            self.encoding_seconds += perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vision-reuse-gate", type=Path)
    parser.add_argument("--frozen-evaluation", type=Path)
    args = parser.parse_args()
    spec, samples = load_spec(args.config), load_samples(args.samples)
    final = (validate_frozen_evaluation(args.frozen_evaluation, args.samples, args.config, "native")
             if args.frozen_evaluation else None)
    if not samples or (not final and any(s.metadata.get("split") not in {"train", "dev"} for s in samples)):
        raise ValueError("This development driver rejects test and unspecified splits.")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Production measurements require a Slurm allocation.")
    if spec.model.device != "cuda" or spec.model.backend != "hf":
        raise ValueError("This driver requires the declared native CUDA model.")
    root = Path(__file__).resolve().parents[1]
    manifest = {"purpose": "development_measurement_not_final_test", "spec": spec.model_dump(),
                "spec_sha256": file_hash(args.config), "samples_sha256": file_hash(args.samples),
                "source_hashes": {str(p.relative_to(root)): file_hash(p)
                                  for d in ("src", "scripts") for p in (root / d).rglob("*.py")},
                "runtime": runtime_manifest(), "slurm_job_id": os.environ["SLURM_JOB_ID"],
                "node": os.environ.get("SLURMD_NODENAME"), "shared_prefill": False,
                "batch_size": 1, "gradient_accumulation": "not_applicable_no_training"}
    manifest["vision_reuse"] = admit_vision_reuse(args.vision_reuse_gate) if args.vision_reuse_gate else None
    manifest["frozen_evaluation"] = final
    if final:
        manifest["purpose"] = "frozen_final_evaluation_no_method_selection"
    create_run(args.output, manifest)
    start = perf_counter()
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no CPU fallback.")
        torch.manual_seed(spec.seed)
        backend = TimedBackend(spec.model, spec.resources, spec.system_prompt, shared_prefill=False,
                               reuse_vision_features=bool(args.vision_reuse_gate))
        backend._ensure_loaded()
        write_json(args.output / "hardware.json", {
            "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
            "driver": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()})
        # Multi-token checks avoid exercising the known-disabled optional single-token shortcut.
        verification_samples = (load_samples(root / final["protocol"]["verification_samples"])
                                if final else samples)
        if final and any(s.metadata["split"] not in {"train", "dev"} for s in verification_samples):
            raise ValueError("Numerical acceptance must use frozen development inputs.")
        verification = verify_native(backend, output=args.output / "native-verification.json",
                                     image_sample=next((s for s in verification_samples if any(p.type == "image" for p in s.parts)), None),
                                     video_sample=next((s for s in verification_samples if any(p.type == "video" for p in s.parts)), None),
                                     anchors=[Anchor(id="multi1", surface="a strong emotion"),
                                              Anchor(id="multi2", surface="an emotional response")])
        if verification["status"] != "passed":
            raise RuntimeError("Native selected/full verification failed.")
        backend.shared_prefill = False
        backend.forward_seconds = backend.encoding_seconds = 0.0
        torch.cuda.reset_peak_memory_stats()
        measured = perf_counter()
        total_items = 0
        for i, sample in enumerate(samples):
            table = measure([sample], spec, backend=backend)
            save_scores(table, args.output / f"part-{i:05d}.parquet")
            append_event(args.output, "sample_completed", sample_id=sample.id,
                         execution=table.manifest["execution"])
            total_items += len(table.frame)
            if table.manifest["execution"]["failed_items"]:
                raise RuntimeError(f"Scoring failed for {sample.id}; partial artifacts retained.")
            print(f"{i + 1}/{len(samples)} {sample.id}", flush=True)
        torch.cuda.synchronize()
        cost = {"measurement_seconds": perf_counter() - measured, "total_seconds": perf_counter() - start,
                "forward_seconds": backend.forward_seconds, "encoding_seconds": backend.encoding_seconds,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "samples": len(samples), "anchors": len(spec.anchors), "bridges": len(spec.bridges),
                "score_items": total_items, "cache_hits": 0}
        write_json(args.output / "cost.json", cost)
        append_event(args.output, "completed", cost=cost)
    except BaseException as exc:
        append_event(args.output, "failed", error=repr(exc), elapsed_seconds=perf_counter() - start)
        raise


if __name__ == "__main__":
    main()
