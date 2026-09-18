"""One YAML, one run directory: measure -> calibrate -> export -> analyze with the library API."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from . import analysis
from .calibration import fit_reference, to_matrix, transform
from .config import resolve_study_config
from .engine import measure
from .io import (load_samples, load_scores, load_spec, save_calibration, save_matrix,
                 save_scores, write_json)
from .provenance import runtime_manifest, stable_hash
from .reliability import audit_reliability
from .types import FrozenModel, ScoreTable, StudySpec, Variant


class ReferenceSpec(FrozenModel):
    """Where the calibration reference comes from. Exactly one source."""
    split: Literal["train", "external"] | None = None
    samples: str | None = None
    scores: str | None = None

    def sources(self) -> list[str]:
        return [name for name in ("split", "samples", "scores") if getattr(self, name) is not None]


class ExportSpec(FrozenModel):
    variant: Variant = "reference_z"
    missing: Literal["error", "drop_samples", "drop_anchors"] = "error"


class AnalysisSpec(FrozenModel):
    kind: Literal["cluster", "pca", "groups", "shift", "network", "reliability"]
    k: int | None = Field(default=None, gt=0)
    network_kind: Literal["concept", "sample"] = "concept"
    bootstrap: int = Field(default=1000, gt=0)
    seed: int = 42


class PipelineSpec(FrozenModel):
    study: str | dict[str, Any]
    samples: str
    output_dir: str
    cache: str | None = None
    reference: ReferenceSpec | None = None
    export: ExportSpec = Field(default_factory=ExportSpec)
    analyses: tuple[AnalysisSpec, ...] = ()


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def load_pipeline(path: str | Path) -> tuple[PipelineSpec, Path]:
    import yaml
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    spec = PipelineSpec.model_validate(raw)
    if spec.reference is not None and len(spec.reference.sources()) != 1:
        raise ValueError("reference needs exactly one of split, samples or scores.")
    if spec.export.variant.startswith("reference") and spec.reference is None:
        raise ValueError(f"export.variant {spec.export.variant!r} requires a reference section.")
    for item in spec.analyses:
        if item.kind == "cluster" and item.k is None:
            raise ValueError("analyses: cluster requires an explicit k.")
    return spec, path.parent


def _study(spec: PipelineSpec, base: Path) -> StudySpec:
    if isinstance(spec.study, str):
        return load_spec(_resolve(base, spec.study))
    return StudySpec.model_validate(resolve_study_config(spec.study, base))


def _subset(table: ScoreTable, ids: set[str]) -> ScoreTable:
    manifest = deepcopy(table.manifest)
    manifest["samples"] = [s for s in manifest["samples"] if s["id"] in ids]
    return ScoreTable(table.frame[table.frame.sample_id.isin(ids)].copy(), manifest)


def _reference_scores(spec: PipelineSpec, base: Path, study: StudySpec, scores: ScoreTable,
                      cache: Path | None) -> tuple[ScoreTable, dict[str, Any]]:
    reference = spec.reference
    assert reference is not None
    if reference.scores is not None:
        path = _resolve(base, reference.scores)
        return load_scores(path), {"source": "scores", "path": str(path)}
    if reference.samples is not None:
        path = _resolve(base, reference.samples)
        table = measure(load_samples(path), study, cache=cache)
        return table, {"source": "samples", "path": str(path), "split": "external"}
    marked = {s["id"] for s in scores.manifest["samples"]
              if str(s.get("metadata", {}).get("split", "")).lower() in {"train", "training"}}
    declared = {s["id"] for s in scores.manifest["samples"] if s.get("metadata", {}).get("split")}
    if reference.split == "train":
        if not marked:
            raise ValueError("reference.split is train but no sample carries metadata.split=train.")
        return _subset(scores, marked), {"source": "study_samples", "split": "train",
                                          "sample_ids": sorted(marked)}
    if declared:
        raise ValueError("reference.split is external but the study samples declare splits; "
                         "supply reference.samples or reference.scores instead.")
    return scores, {"source": "study_samples", "split": "external",
                    "note": "All study samples were used as the reference; declare splits "
                            "or supply external reference material for held-out evaluation."}


def run_pipeline(path: str | Path) -> dict[str, Any]:
    """Execute a pipeline YAML; every artifact lands in ``output_dir`` with a run record."""
    spec, base = load_pipeline(path)
    study = _study(spec, base)
    samples = load_samples(_resolve(base, spec.samples))
    output = _resolve(base, spec.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache = _resolve(base, spec.cache) if spec.cache else None
    record: dict[str, Any] = {
        "pipeline": spec.model_dump(mode="json"), "pipeline_path": str(Path(path).resolve()),
        "study": study.model_dump(mode="json"), "study_id": stable_hash(study.model_dump(mode="json")),
        "samples": len(samples), "runtime": runtime_manifest(), "artifacts": {},
    }
    scores = measure(samples, study, cache=cache)
    save_scores(scores, output / "scores.parquet")
    record["artifacts"]["scores"] = "scores.parquet"
    record["execution"] = scores.manifest["execution"]
    table = scores
    if spec.reference is not None:
        reference_scores, provenance = _reference_scores(spec, base, study, scores, cache)
        split = None if provenance.get("source") == "scores" else provenance.get("split")
        artifact = fit_reference(reference_scores, split=split)
        save_calibration(artifact, output / "reference.json")
        table = transform(scores, artifact)
        save_scores(table, output / "calibrated.parquet")
        record["reference"] = {**provenance, "coordinates": len(artifact.statistics),
                               "undefined_z": int((artifact.statistics.z_status != "ok").sum())}
        record["artifacts"].update(reference="reference.json", calibrated="calibrated.parquet")
    matrix = to_matrix(table, variant=spec.export.variant, missing=spec.export.missing)
    save_matrix(matrix, output / "matrix.npz")
    record["artifacts"]["matrix"] = "matrix.npz"
    record["matrix"] = {"shape": list(matrix.values.shape), "variant": matrix.variant,
                        "coverage": matrix.manifest.get("coverage")}
    for item in spec.analyses:
        name = item.kind if item.kind != "network" else f"network-{item.network_kind}"
        if item.kind == "reliability":
            result = audit_reliability(scores)
        elif item.kind == "cluster":
            result = analysis.cluster(matrix, k=item.k, seed=item.seed)
        elif item.kind == "pca":
            result = analysis.pca(matrix)
        elif item.kind == "groups":
            result = analysis.compare_groups(matrix, n_bootstrap=item.bootstrap, seed=item.seed)
        elif item.kind == "shift":
            result = analysis.semantic_shift(matrix, n_bootstrap=item.bootstrap, seed=item.seed)
        else:
            kwargs = {"k": item.k or 5} if item.network_kind == "sample" else {}
            result = analysis.semantic_network(matrix, kind=item.network_kind, **kwargs)
        write_json(output / f"analysis-{name}.json", result)
        record["artifacts"][f"analysis-{name}"] = f"analysis-{name}.json"
    write_json(output / "run.json", record)
    return {"output_dir": str(output), **record["execution"], "artifacts": record["artifacts"]}
