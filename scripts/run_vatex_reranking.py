"""Official caption-to-video reranking with explicit candidate support and no gold reads."""
import argparse
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from analyze_vatex import align_scores, two_stage_priority
from frozen_evaluation import validate_frozen_evaluation
from vision_reuse_admission import admit_vision_reuse
from omnianchor.backends.vision_reuse import reuse_vision_outputs
from omnianchor.campaign import append_event, create_run
from omnianchor.io import load_samples, load_spec, read_json, write_json
from omnianchor.official_baselines import QwenRetrieval
from omnianchor.provenance import file_hash, runtime_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["samples", "config", "vision-reuse-gate", "output"]:
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--embedding-scores", type=Path)
    parser.add_argument("--frozen-evaluation", type=Path)
    parser.add_argument("--video-shard", type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Direct reranking requires a Slurm GPU allocation.")
    samples = load_samples(args.samples)
    final = validate_frozen_evaluation(args.frozen_evaluation, args.samples, args.config, "qwen-reranker") if args.frozen_evaluation else None
    if not final and any(s.metadata["split"] not in {"train", "dev"} for s in samples):
        raise ValueError("Unadmitted protected inputs.")
    if len({s.id for s in samples}) != len(samples) or any(len(s.parts) != 1 for s in samples):
        raise ValueError("Video and caption must be distinct uniquely identified inputs.")
    videos = sorted([s for s in samples if s.parts[0].type == "video"], key=lambda s: s.id)
    queries = sorted([s for s in samples if s.parts[0].type == "text"], key=lambda s: s.id)
    if not videos or not queries or len(videos)+len(queries) != len(samples):
        raise ValueError("Require videos and separate text captions only.")
    video_ids, query_ids = [s.id for s in videos], [s.id for s in queries]
    owned_ids = read_json(args.video_shard) if args.video_shard else video_ids
    if (not isinstance(owned_ids, list) or not owned_ids or len(set(owned_ids)) != len(owned_ids)
            or not set(owned_ids) <= set(video_ids)):
        raise ValueError("Video shard must contain unique IDs from the full gallery.")
    if args.video_shard and (args.pilot or (final and file_hash(args.video_shard) != final["protocol"]["video_shard_sha256"])):
        raise ValueError("Pilot cannot be sharded; final shard must match its admission.")
    if not {s.pair_id for s in queries} <= set(video_ids) or any(not s.parts[0].text.strip() for s in queries):
        raise ValueError("Caption pair IDs or text are invalid.")
    if args.pilot:
        if final or args.embedding_scores or len(samples) != 32:
            raise ValueError("Pilot is exactly32development inputs with all video-caption pairs scored.")
        stage_one = np.zeros((len(queries), len(videos)))
        selected = np.ones_like(stage_one, dtype=bool)
    else:
        if not args.embedding_scores or len({s.language for s in queries}) != 1:
            raise ValueError("Full reranking requires original embedding scores and one caption language.")
        if final and file_hash(args.embedding_scores) != final["protocol"]["embedding_scores_sha256"]:
            raise ValueError("Frozen stage-one scores changed.")
        stage_one = align_scores(args.embedding_scores, query_ids, video_ids)
        order = np.argsort(-stage_one, axis=1, kind="stable")
        selected = np.zeros_like(stage_one, dtype=bool)
        np.put_along_axis(selected, order[:, :min(100, len(videos))], True, axis=1)
    spec = load_spec(args.config)
    root = Path(__file__).resolve().parents[1]
    model_id = "Qwen/Qwen3-VL-Reranker-2B"
    revision = read_json(root/"configs/models-20260910.json")["models"][model_id]
    instruction = "Retrieve the video described by the query."
    create_run(args.output, {"purpose": "32input technical direct-reranker cost pilot" if args.pilot else "Fixed embedding-top100 caption-to-video reranking",
        "samples_sha256": file_hash(args.samples), "resource_spec_sha256": file_hash(args.config),
        "model_id": model_id, "revision": revision, "instruction": instruction, "resources": spec.resources.model_dump(),
        "runtime": runtime_manifest(), "seed": 42, "pilot": args.pilot, "frozen_evaluation": final,
        "stage_one_sha256": file_hash(args.embedding_scores) if args.embedding_scores else None,
        "query_ids": query_ids, "video_ids": video_ids, "selected_pairs": int(selected.sum()),
        "owned_video_ids": owned_ids,
        "video_shard_sha256": file_hash(args.video_shard) if args.video_shard else None,
        "candidate_rule": "all pairs for technicalpilot; fullrun topmin(100,gallerysize) original embedding cosines percaption, stable lexicographic video-ID ties",
        "vision_reuse": admit_vision_reuse(args.vision_reuse_gate), "media_preprocessing_reuse": False,
        "score_units": "Raw selected yesprobabilities with NaN outside support; retrieval priority2+yes within support and original embedding score outside. Priority is not a probability.",
        "direction": "caption_to_video only; transpose is not a valid reverse-reranking protocol",
        "input_policy": "Video inputs contain no captions. Query is the candidate caption, never human relevance labels.",
        "source_sha256": {p: file_hash(root/p) for p in ["scripts/run_vatex_reranking.py", "scripts/analyze_vatex.py", "src/omnianchor/official_baselines.py"]}})
    started = perf_counter()
    try:
        import torch
        if not torch.cuda.is_available() or "5000 Ada" not in torch.cuda.get_device_name():
            raise RuntimeError("Require the frozen RTX5000Ada GPU class; no CPU fallback.")
        torch.manual_seed(42)
        model = QwenRetrieval(model_id, revision, root/"research/upstream/qwen3-vl-embedding-393e297", spec.resources)
        torch.cuda.reset_peak_memory_stats()
        measured = perf_counter()
        probabilities = np.full(stage_one.shape, np.nan)
        for j, video in enumerate(videos):
            if video.id not in owned_ids:
                continue
            index = np.flatnonzero(selected[:, j])
            if len(index):
                with reuse_vision_outputs(model.model):
                    values = model.score(video, [queries[i].parts[0].text for i in index], instruction)
                if not np.isfinite(values).all():
                    raise ValueError("Nonfinite selected reranking scores.")
                probabilities[index, j] = values
            np.savez_compressed(args.output/f"part-{j:05d}.npz", video_id=video.id,
                query_ids=np.array(query_ids)[index], probabilities=probabilities[index, j])
            append_event(args.output, "video_completed", video_id=video.id, selected_queries=len(index))
            print(j+1, len(videos), video.id, len(index), flush=True)
        if args.video_shard:
            columns = [video_ids.index(i) for i in owned_ids]
            np.savez_compressed(args.output/"shard.npz", probabilities=probabilities[:, columns],
                selected=selected[:, columns], query_ids=np.array(query_ids), candidate_ids=np.array(owned_ids))
        else:
            priority = two_stage_priority(stage_one, probabilities, selected)
            np.savez_compressed(args.output/"matrix.npz", scores=priority, probabilities=probabilities,
                selected=selected, embedding_scores=stage_one if args.embedding_scores else np.empty((0, 0)),
                query_ids=np.array(query_ids), candidate_ids=np.array(video_ids))
        torch.cuda.synchronize()
        write_json(args.output/"cost.json", {"measurement_seconds": perf_counter()-measured,
            "total_seconds": perf_counter()-started, "videos": len(videos), "captions": len(queries),
            "pairs": int(selected[:, [video_ids.index(i) for i in owned_ids]].sum()),
            "owned_videos": len(owned_ids), "gpu": torch.cuda.get_device_name(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
        append_event(args.output, "completed")
    except BaseException as exc:
        append_event(args.output, "failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
