"""Bounded official FLA versus PyTorch kernel equivalence gate; no KV reuse or budget change."""
import argparse
import inspect
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from omnianchor import measure
from omnianchor.backends.hf import HFBackend
from omnianchor.campaign import append_event, create_run
from omnianchor.io import load_samples, load_spec, save_scores, write_json
from omnianchor.types import Anchor
from verify_native import verify_native


class KernelBackend(HFBackend):
    kernel_path = "official_fla"

    @property
    def identity(self):
        return {**super().identity, "linear_attention_kernel_gate_path": self.kernel_path}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Require Slurm GPU allocation.")
    root = Path(__file__).resolve().parents[1]
    spec = load_spec(root / "configs/smoke/media-qwen35.json")
    spec = spec.model_copy(update={"anchors": (Anchor(id="a", surface="joy"), Anchor(id="b", surface="sadness"),
        Anchor(id="c", surface="a strong emotion")), "bridges": spec.bridges[:1]})
    samples = []
    for dataset in ["emobank-reader", "chinese-emobank-sentence"]:
        samples.extend(load_samples(root / "data/prepared/text-smoke-20260910-02" / dataset / "smoke-samples.json"))
    samples.extend(load_samples(root / "data/prepared/multimodal-smoke-20260910-02/samples.json"))
    if len(samples) != 40 or any(s.metadata["split"] not in {"train", "dev"} for s in samples):
        raise ValueError("Require frozen32text+8media development inputs.")
    create_run(args.output, {"purpose": "kernel equivalence and cost; not construct validity",
        "spec": spec.model_dump(), "sample_ids": [s.id for s in samples], "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "threshold_nat": .01, "shared_prefill": False, "comparison": "same model/inputs, only chunk gated-delta kernel changes"})
    try:
        import torch
        from transformers.models.qwen3_5 import modeling_qwen3_5 as module
        torch.manual_seed(42)
        accelerated = module.torch_chunk_gated_delta_rule
        reference = inspect.unwrap(accelerated)
        wrappers = []
        current = accelerated
        while current is not None:
            closure = inspect.getclosurevars(current).nonlocals
            implementation = closure.get("implementation")
            if implementation is not None:
                wrappers.append({"module": implementation.__module__, "name": implementation.__name__})
            current = getattr(current, "__wrapped__", None)
        if not any(w["module"].startswith("fla.") for w in wrappers):
            raise RuntimeError("Official FLA kernel is not selected; no silent fallback benchmark.")
        write_json(args.output / "kernel-selection.json", {"wrappers": wrappers,
            "reference_module": reference.__module__, "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda})
        backend = KernelBackend(spec.model, spec.resources, shared_prefill=False)
        backend._ensure_loaded()
        records = []
        for i, sample in enumerate(samples):
            results = {}
            for name, function in [("pytorch_reference", reference), ("official_fla", accelerated)]:
                backend.kernel_path = name
                module.torch_chunk_gated_delta_rule = function
                torch.cuda.synchronize()
                start = perf_counter()
                table = measure([sample], spec, backend=backend)
                torch.cuda.synchronize()
                seconds = perf_counter()-start
                save_scores(table, args.output / name / f"part-{i:05d}.parquet")
                if table.manifest["execution"]["failed_items"]:
                    raise RuntimeError("Scoring failed; reject acceleration.")
                results[name] = (table.frame.set_index("anchor_id").loc[[a.id for a in spec.anchors]], seconds)
            slow, fast = results["pytorch_reference"][0], results["official_fla"][0]
            if slow.token_ids.tolist() != fast.token_ids.tolist():
                raise RuntimeError("Kernel comparison tokenization differs.")
            error = float(np.max(np.abs(slow.raw_logp.to_numpy()-fast.raw_logp.to_numpy())))
            token_error = max(abs(a-b) for slow_tokens, fast_tokens in zip(slow.token_logps, fast.token_logps)
                              for a, b in zip(slow_tokens, fast_tokens))
            record = {"sample_id": sample.id, "modality": sample.parts[0].type, "max_abs_sum_nat": error,
                "max_abs_token_nat": token_error, "reference_seconds": results["pytorch_reference"][1],
                "fla_seconds": results["official_fla"][1], "within_threshold": max(error, token_error) <= .01}
            records.append(record)
            append_event(args.output, "comparison", **record)
            print(record, flush=True)
        module.torch_chunk_gated_delta_rule = accelerated
        backend.kernel_path = "official_fla"
        verification = verify_native(backend, output=args.output / "native-verification.json",
            image_sample=next(s for s in samples if s.parts[0].type == "image"),
            video_sample=next(s for s in samples if s.parts[0].type == "video"),
            anchors=[Anchor(id="m1", surface="a strong emotion"), Anchor(id="m2", surface="an emotional response")])
        passed = all(r["within_threshold"] for r in records) and verification["status"] == "passed"
        write_json(args.output / "summary.json", {"status": "passed" if passed else "failed",
            "comparisons": records, "threshold_nat": .01, "jit_note": "first FLA invocation includes compilation",
            "authorization_to_mix_existing_scores": False, "requires_new_measurement_id": True})
        append_event(args.output, "completed" if passed else "gate_failed")
        if not passed:
            raise RuntimeError("Numerical gate failed: acceleration remains disabled for production.")
    except BaseException as exc:
        append_event(args.output, "failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
