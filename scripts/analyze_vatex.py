"""Evaluate development VATEX retrieval matrices against original paired video IDs."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from vlanchor.campaign import append_event, create_run
from vlanchor.evaluation import group_bootstrap, retrieval_metrics
from vlanchor.io import load_samples, write_json
from vlanchor.provenance import file_hash


def two_stage_priority(embedding_scores, probabilities, selected):
    """Rank reranked candidates first, then untouched candidates by original embedding."""
    embedding = np.asarray(embedding_scores, dtype=float)
    probabilities, selected = np.asarray(probabilities, dtype=float), np.asarray(selected, dtype=bool)
    if embedding.shape != probabilities.shape or embedding.shape != selected.shape or embedding.ndim != 2:
        raise ValueError("Two-stage retrieval matrices must align.")
    if not np.isfinite(embedding).all() or np.any(np.abs(embedding) > 1+1e-5):
        raise ValueError("Stage one must be complete original embedding cosine scores.")
    if not np.isfinite(probabilities[selected]).all() or np.any((probabilities[selected] < 0) | (probabilities[selected] > 1)):
        raise ValueError("Missing or invalid reranker probability for selected pairs.")
    if not np.isnan(probabilities[~selected]).all():
        raise ValueError("Unscored probabilities must remain undefined, not be filled.")
    if np.any(selected.sum(axis=1) == 0):
        raise ValueError("Every query needs its selected candidate set.")
    return np.where(selected, 2+probabilities, embedding)


def retrieval_rows(relevance, scores):
    """Retain query-level contributions for paired, source-group uncertainty."""
    summary = retrieval_metrics(relevance, scores)
    if summary["excluded_queries"]:
        raise ValueError("Missing paired item in the fixed gallery; no query exclusion.")
    order = np.argsort(-scores, axis=1, kind="stable")
    ranked = np.take_along_axis(np.asarray(relevance), order, axis=1)
    rows = {}
    for k in (1, 5, 10):
        found = ranked[:, :k].sum(axis=1)
        rows[f"success@{k}"] = (found > 0).astype(float)
        rows[f"recall@{k}"] = found / np.asarray(relevance).sum(axis=1)
    if any(not np.isclose(values.mean(), summary[name]) for name, values in rows.items()):
        raise AssertionError("Query contributions disagree with the shared evaluation API.")
    return rows


def align_scores(path, queries, candidates):
    with np.load(path, allow_pickle=False) as data:
        query_ids, candidate_ids = data["query_ids"].tolist(), data["candidate_ids"].tolist()
        scores = data["scores"].astype(float)
    if (len(set(query_ids)) != len(query_ids) or len(set(candidate_ids)) != len(candidate_ids)
            or set(query_ids) != set(queries) or set(candidate_ids) != set(candidates)):
        raise ValueError("Retrieval IDs differ from the full frozen population.")
    if scores.shape != (len(query_ids), len(candidate_ids)) or not np.isfinite(scores).all():
        raise ValueError("Require complete finite retrieval scores, including unselected candidates.")
    qi, ci = {v: i for i, v in enumerate(query_ids)}, {v: i for i, v in enumerate(candidate_ids)}
    return scores[np.ix_([qi[i] for i in queries], [ci[i] for i in candidates])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--captions", type=Path, required=True)
    parser.add_argument("--scores", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--caption-only", action="store_true")
    args = parser.parse_args()
    videos, captions = load_samples(args.videos), load_samples(args.captions)
    samples = videos + captions
    if not videos or not captions or any(s.metadata.get("split") != "dev" for s in samples):
        raise ValueError("Development only; protected evaluation requires a separate frozen admission.")
    if any(len(s.parts) != 1 or s.parts[0].type != "video" for s in videos):
        raise ValueError("Gallery videos must not contain retrieval captions.")
    if any(len(s.parts) != 1 or s.parts[0].type != "text" for s in captions):
        raise ValueError("Caption queries must contain text only.")
    if len({s.language for s in captions}) != 1:
        raise ValueError("Evaluate English and Chinese separately.")
    videos, captions = sorted(videos, key=lambda s: s.id), sorted(captions, key=lambda s: s.id)
    vi, ci = [s.id for s in videos], [s.id for s in captions]
    if len(set(vi+ci)) != len(vi+ci):
        raise ValueError("Duplicate sample IDs.")
    vp, cp = vi, [s.pair_id for s in captions]
    if None in vp+cp or len(set(vp)) != len(vp) or set(vp) != set(cp):
        raise ValueError("Original video-caption pair IDs must identify exactly one gallery video.")
    relevance = np.equal.outer(cp, vp)
    if not np.all(relevance.sum(axis=0) == 10):
        raise ValueError("Require all ten published captions for every selected video.")
    groups = [s.group_id for s in captions]
    if any(not g for g in groups) or any(not s.group_id for s in videos):
        raise ValueError("Require YouTube source groups; captions are not independent observations.")
    source_by_pair = dict(zip(vp, [s.group_id for s in videos]))
    if any(s.group_id != source_by_pair[s.pair_id] for s in captions):
        raise ValueError("Caption and paired video source groups disagree.")
    if len({p.stem for p in args.scores}) != len(args.scores):
        raise ValueError("Method filenames must be unique.")
    create_run(args.output, {"purpose": "VATEX dev retrieval, no method selection or test access",
        "inputs": {str(p): file_hash(p) for p in [args.videos, args.captions, *args.scores]},
        "script_sha256": file_hash(Path(__file__)), "query_ids": ci, "candidate_ids": vi,
        "relevance": "original paired video; unannotated alternative matches remain a limitation",
        "tie_policy": "stable lexicographic candidate sample ID",
        "uncertainty": "1000 source-YouTube-group bootstrap draws, fixed gallery; captions co-resampled",
        "inference_scope": "development descriptive intervals; no confirmatory p-values or FDR family",
        "primary_direction": "caption_to_video", "secondary_direction": None if args.caption_only else "video_to_caption"})
    try:
        metrics, per_query, methods = [], [], {}
        for path in args.scores:
            scores = align_scores(path, ci, vi)
            directions = [
                ("caption_to_video", relevance, scores, ci, groups),
                ("video_to_caption", relevance.T, scores.T, vi, [s.group_id for s in videos]),
            ]
            for direction, y, x, ids, units in directions[:1] if args.caption_only else directions:
                contributions = retrieval_rows(y, x)
                methods[path.stem, direction] = contributions
                for metric, values in contributions.items():
                    interval = group_bootstrap(lambda index: values[index].mean(), units)
                    metrics.append(dict(method=path.stem, direction=direction, metric=metric, **interval))
                    per_query.extend(dict(method=path.stem, direction=direction, metric=metric,
                        sample_id=i, source_group=g, value=float(v)) for i, g, v in zip(ids, units, values))
        paired = []
        names = [p.stem for p in args.scores]
        for a_index, a in enumerate(names):
            for b in names[a_index+1:]:
                directions = [("caption_to_video", groups), ("video_to_caption", [s.group_id for s in videos])]
                for direction, units in directions[:1] if args.caption_only else directions:
                    for metric in methods[a, direction]:
                        difference = methods[a, direction][metric] - methods[b, direction][metric]
                        interval = group_bootstrap(lambda index: difference[index].mean(), units)
                        paired.append(dict(method_a=a, method_b=b, direction=direction, metric=metric, **interval))
        pd.DataFrame(metrics).to_csv(args.output/"metrics.csv", index=False)
        pd.DataFrame(per_query).to_csv(args.output/"per-query.csv", index=False)
        pd.DataFrame(paired).to_csv(args.output/"paired-differences.csv", index=False)
        write_json(args.output/"coverage.json", {"videos": len(vi), "captions": len(ci),
            "source_groups": len(set(groups)), "excluded_queries": [], "test_used": False})
        append_event(args.output, "completed")
    except BaseException as exc:
        append_event(args.output, "failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
