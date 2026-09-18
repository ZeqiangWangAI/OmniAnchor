"""Read-only compute profiling on frozen development inputs; no scoring optimization."""
import argparse
import cProfile
import io
import os
import pstats
from pathlib import Path

from omnianchor.backends.hf import HFBackend
from omnianchor.campaign import create_run
from omnianchor.io import load_samples, load_spec, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Require Slurm allocation.")
    root = Path(__file__).resolve().parents[1]
    spec = load_spec(root / "configs/smoke/media-qwen35.json")
    media = load_samples(root / "data/prepared/multimodal-smoke-20260910-02/samples.json")
    texts = load_samples(root / "data/prepared/text-smoke-20260910-02/emobank-reader/smoke-samples.json")
    samples = [texts[0], next(s for s in media if s.parts[0].type=="image"), next(s for s in media if s.parts[0].type=="video")]
    create_run(args.output, {"purpose": "profile only; not a new measurement method", "spec": spec.model_dump(),
        "sample_ids": [s.id for s in samples], "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "environment": {k: os.environ.get(k) for k in ["CUDA_VISIBLE_DEVICES", "CUDA_LAUNCH_BLOCKING", "OMP_NUM_THREADS"]}})
    import torch
    backend = HFBackend(spec.model, spec.resources, shared_prefill=False)
    backend._ensure_loaded()
    write_json(args.output / "model-placement.json", {"devices": sorted({str(p.device) for p in backend._model.parameters()}),
        "dtypes": sorted({str(p.dtype) for p in backend._model.parameters()}), "gpu": torch.cuda.get_device_name()})
    for sample in samples:
        modality = sample.parts[0].type
        torch.cuda.synchronize()
        profile = cProfile.Profile()
        profile.enable()
        result = backend.score_candidates(sample, spec.bridges[0], spec.anchors[:2])
        torch.cuda.synchronize()
        profile.disable()
        profile.dump_stats(args.output / f"{modality}.pstats")
        stream = io.StringIO()
        pstats.Stats(profile, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(70)
        (args.output / f"{modality}-profile.txt").write_text(stream.getvalue())
        write_json(args.output / f"{modality}-scores.json", result)


if __name__ == "__main__":
    main()
