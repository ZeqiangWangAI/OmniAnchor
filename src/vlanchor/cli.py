"""Command-line entrypoints use the same Python measurement APIs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from . import analysis
from .calibration import fit_reference, to_matrix, transform
from .engine import measure
from .io import (jsonable, load_calibration, load_matrix, load_samples, load_scores, load_spec,
                 read_json, save_calibration, save_matrix, save_scores, write_json)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vlanchor", description="Traceable probabilistic anchor measurement")
    sub = p.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="Validate study and optional local samples without loading weights")
    validate.add_argument("--config", required=True)
    validate.add_argument("--samples")
    prepare = sub.add_parser("prepare", help="Convert an explicitly supplied local dataset to samples and labels")
    prepare.add_argument("--config", required=True, help="YAML with dataset and adapter_kwargs")
    prepare.add_argument("--output", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--config", required=True)
    m.add_argument("--samples", required=True)
    m.add_argument("--output", required=True)
    m.add_argument("--cache")
    cal = sub.add_parser("calibrate")
    cal.add_argument("--reference-scores", required=True)
    cal.add_argument("--output", required=True, help="Calibration JSON")
    cal.add_argument("--reference-split", choices=["train", "external"])
    cal.add_argument("--scores", help="Optional score table to transform")
    cal.add_argument("--transformed-output")
    ex = sub.add_parser("export")
    ex.add_argument("--scores", required=True)
    ex.add_argument("--calibration")
    ex.add_argument("--variant", default="raw_logp", choices=["raw_logp", "mean_token_logp", "reference_log_ratio", "reference_z"])
    ex.add_argument("--missing", default="error", choices=["error", "drop_samples", "drop_anchors"])
    ex.add_argument("--output", required=True)
    an = sub.add_parser("analyze")
    an.add_argument("kind", choices=["cluster", "pca", "groups", "shift", "network", "reliability"])
    an.add_argument("--input", required=True, help="Matrix, or scores for reliability")
    an.add_argument("--output", required=True)
    an.add_argument("--k", type=int)
    an.add_argument("--metadata", help="JSON records with id, target_id/time/group_id")
    an.add_argument("--network-kind", choices=["concept", "sample"], default="concept")
    an.add_argument("--bootstrap", type=int, default=1000)
    an.add_argument("--seed", type=int, default=42)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--kind", choices=["multilabel", "vad", "candidates", "retrieval"], required=True)
    ev.add_argument("--predictions", required=True)
    ev.add_argument("--labels", required=True)
    ev.add_argument("--output", required=True)
    op = sub.add_parser("optimize-bridges", help="Bounded subset selection from a pre-scored training bridge pool")
    op.add_argument("--scores", required=True)
    op.add_argument("--labels", required=True, help="CSV indexed by sample_id with anchor columns")
    op.add_argument("--output", required=True)
    op.add_argument("--variant", default="reference_z")
    op.add_argument("--split", choices=["train"], required=True)
    op.add_argument("--min-class-count", type=int, default=5)
    return p


def execute(args) -> dict:
    if args.command == "validate":
        spec = load_spec(args.config)
        samples = load_samples(args.samples) if args.samples else []
        return {"valid": True, "study": spec.name, "anchors": len(spec.anchors),
                "bridges": len(spec.bridges), "samples": len(samples),
                "model_weights_loaded": False}
    if args.command == "prepare":
        from .datasets import load_dataset
        cfg_path = Path(args.config).resolve()
        cfg = yaml.safe_load(cfg_path.read_text())
        kwargs = dict(cfg.get("adapter_kwargs", {}))
        for key, value in kwargs.items():
            if isinstance(value, str) and (key == "path" or key.endswith(("_path", "_dir", "_root"))):
                kwargs[key] = str((cfg_path.parent / value).resolve())
        bundle = load_dataset(cfg["dataset"], **kwargs)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "samples.json", bundle.samples)
        labels = bundle.labels.reset_index() if any(bundle.labels.index.names) else bundle.labels
        labels.to_csv(output / "labels.csv", index=False)
        write_json(output / "dataset.manifest.json", bundle.manifest)
        if bundle.candidates:
            write_json(output / "candidates.json", bundle.candidates)
        return {"samples": len(bundle.samples), "output": str(output)}
    if args.command == "measure":
        scores = measure(load_samples(args.samples), load_spec(args.config), cache=args.cache)
        save_scores(scores, args.output)
        return {"output": args.output, **scores.manifest["execution"]}
    if args.command == "calibrate":
        if bool(args.scores) != bool(args.transformed_output):
            raise ValueError("--scores and --transformed-output must be supplied together.")
        artifact = fit_reference(load_scores(args.reference_scores), split=args.reference_split)
        save_calibration(artifact, args.output)
        if args.scores:
            save_scores(transform(load_scores(args.scores), artifact), args.transformed_output)
        return {"output": args.output, "coordinates": len(artifact.statistics)}
    if args.command == "export":
        scores = load_scores(args.scores)
        if args.calibration:
            scores = transform(scores, load_calibration(args.calibration))
        matrix = to_matrix(scores, variant=args.variant, missing=args.missing)
        save_matrix(matrix, args.output)
        return {"output": args.output, "shape": matrix.values.shape}
    if args.command == "analyze":
        if args.kind == "reliability":
            from .reliability import audit_reliability
            result = audit_reliability(load_scores(args.input))
        else:
            matrix = load_matrix(args.input)
            meta = read_json(args.metadata) if args.metadata else None
            if args.kind == "cluster":
                if args.k is None:
                    raise ValueError("Clustering requires an explicit --k.")
                result = analysis.cluster(matrix, k=args.k, seed=args.seed)
            elif args.kind == "pca":
                result = analysis.pca(matrix)
            elif args.kind == "groups":
                groups = {r["id"]: r["group_id"] for r in meta} if meta else None
                result = analysis.compare_groups(matrix, groups, n_bootstrap=args.bootstrap, seed=args.seed)
            elif args.kind == "shift":
                result = analysis.semantic_shift(matrix, meta, n_bootstrap=args.bootstrap, seed=args.seed)
            else:
                kwargs = {"k": args.k or 5} if args.network_kind == "sample" else {}
                result = analysis.semantic_network(matrix, kind=args.network_kind, **kwargs)
        write_json(args.output, result)
        return {"output": args.output}
    if args.command == "evaluate":
        from . import evaluation
        pred, labels = pd.read_csv(args.predictions), pd.read_csv(args.labels)
        if args.kind in ("multilabel", "vad"):
            if "sample_id" not in pred or "sample_id" not in labels:
                raise ValueError("Prediction and label CSVs require sample_id.")
            pred, labels = pred.set_index("sample_id"), labels.set_index("sample_id")
            if pred.index.has_duplicates or labels.index.has_duplicates:
                raise ValueError("Duplicate evaluation IDs.")
            if set(pred.index) != set(labels.index) or set(pred.columns) != set(labels.columns):
                raise ValueError("Predictions and labels must have identical samples and targets.")
            labels = labels.loc[pred.index, pred.columns]
            fn = evaluation.evaluate_multilabel if args.kind == "multilabel" else evaluation.evaluate_vad
            name_kwarg = "anchor_ids" if args.kind == "multilabel" else "dimensions"
            result = fn(labels.to_numpy(), pred.to_numpy(), **{name_kwarg: list(pred.columns)})
        elif args.kind == "retrieval":
            id_column = next((c for c in ("query_id", "sample_id") if c in pred or c in labels), None)
            if id_column:
                if id_column not in pred or id_column not in labels:
                    raise ValueError("Retrieval query IDs must be present in both CSVs.")
                pred, labels = pred.set_index(id_column), labels.set_index(id_column)
            if pred.index.has_duplicates or labels.index.has_duplicates:
                raise ValueError("Duplicate retrieval query IDs.")
            if set(pred.index) != set(labels.index) or set(pred.columns) != set(labels.columns):
                raise ValueError("Retrieval predictions and labels must have identical queries and candidates.")
            labels = labels.loc[pred.index, pred.columns]
            result = evaluation.retrieval_metrics(labels.to_numpy(), pred.to_numpy())
            result["query_alignment"] = id_column or "explicit_csv_row_order"
        else:
            keys = ["sample_id", "anchor_id"]
            if "relevant" in labels and "label" not in labels:
                labels = labels.rename(columns={"relevant": "label"})
            if not set(keys + ["score"]).issubset(pred) or not set(keys + ["label"]).issubset(labels):
                raise ValueError("Candidate CSVs require sample_id,anchor_id and score/label.")
            if pred.duplicated(keys).any() or labels.duplicated(keys).any():
                raise ValueError("Duplicate candidate IDs.")
            merged = pred.merge(labels, on=keys, how="outer", validate="one_to_one", indicator=True)
            if merged._merge.ne("both").any():
                raise ValueError("Candidate labels/scores must have identical keys.")
            ordered = merged.sort_values(keys)
            y = {s: group.label.tolist() for s, group in ordered.groupby("sample_id")}
            scores = {s: group.score.tolist() for s, group in ordered.groupby("sample_id")}
            result = evaluation.evaluate_candidates(y, scores)
        write_json(args.output, result)
        return {"output": args.output}
    if args.command == "optimize-bridges":
        from .optimization import OptimizationConfig, OptimizationData, optimize_bridges
        from .types import Bridge
        table = load_scores(args.scores)
        bridges = tuple(Bridge.model_validate(b) for b in table.manifest["bridges"])
        sample_ids = tuple(s["id"] for s in table.manifest["samples"])
        anchor_ids = tuple(a["id"] for a in table.manifest["anchors"])
        labels = pd.read_csv(args.labels)
        if "sample_id" not in labels or not set(anchor_ids).issubset(labels.columns):
            raise ValueError("Optimization labels require sample_id and every anchor column.")
        labels = labels.set_index("sample_id")
        if labels.index.has_duplicates or set(labels.index) != set(sample_ids):
            raise ValueError("Optimization labels must contain exactly the unique score sample IDs.")
        if args.variant not in table.frame:
            raise ValueError(f"Score variant {args.variant!r} is absent; apply calibration first.")
        observed_splits = {s.get("metadata", {}).get("split") for s in table.manifest["samples"]}
        if observed_splits - {None, "train"}:
            raise ValueError("Score table contains non-training samples.")
        data = OptimizationData(labels.loc[list(sample_ids), list(anchor_ids)].to_numpy(),
                                sample_ids, anchor_ids, split=args.split)
        def scorer(bridge):
            frame = table.frame[table.frame.bridge_id == bridge.id]
            if frame.status.ne("ok").any() or frame.duplicated(["sample_id", "anchor_id"]).any():
                raise ValueError("Optimization requires complete successful score rows.")
            return frame.pivot(index="sample_id", columns="anchor_id", values=args.variant).loc[
                list(sample_ids), list(anchor_ids)].to_numpy()
        result = optimize_bridges(bridges, data, OptimizationConfig(
            rounds=0, max_unique_bridges=16, min_positive=args.min_class_count,
            min_negative=args.min_class_count,
            forbidden_surfaces=tuple(a["surface"] for a in table.manifest["anchors"])),
            score_bridge=scorer)
        write_json(args.output, result)
        return {"output": args.output, "status": result.status}
    raise ValueError("Unknown command.")


def main(argv=None) -> int:
    import json
    p = parser()
    try:
        result = execute(p.parse_args(argv))
    except (ValueError, FileNotFoundError) as exc:
        p.exit(2, f"vlanchor: {exc}\n")
    print(json.dumps(jsonable(result), ensure_ascii=False, allow_nan=False))
    return 1 if result.get("failed_items", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
