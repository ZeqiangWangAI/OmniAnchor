"""Reference calibration without changing the definition of anchor events."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from .errors import IncompatibleMeasurement, MissingScores
from .types import CalibrationArtifact, MeasurementMatrix, ScoreTable

_KEYS = ["sample_id", "anchor_id", "bridge_id"]
_VARIANTS = {"raw_logp", "mean_token_logp", "reference_log_ratio", "reference_z"}


def _configuration(scores: ScoreTable) -> dict[str, Any]:
    """Use exact instrument/coordinate definitions, excluding input identities."""
    frame = scores.frame
    required = _KEYS + ["anchor_surface", "bridge_prefix", "relation", "event",
                        "raw_logp", "status", "model_id", "model_revision"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise MissingScores(f"Score table lacks columns: {sorted(missing)}")
    if frame[_KEYS].isna().any().any() or frame.duplicated(_KEYS).any():
        raise MissingScores("Score identities must be non-null and unique.")
    if frame.empty:
        raise MissingScores("The score table is empty.")
    for columns in (["anchor_id", "anchor_surface"],
                    ["bridge_id", "bridge_prefix", "relation"]):
        unique = frame[columns].drop_duplicates()
        if unique[columns[0]].duplicated().any():
            raise IncompatibleMeasurement(f"Inconsistent definitions for {columns[0]}.")
    for column in ["relation", "event", "model_id", "model_revision"]:
        if frame[column].nunique(dropna=False) != 1:
            raise IncompatibleMeasurement(f"Mixed {column} values are not one measurement.")
    manifest = scores.manifest
    anchors = manifest.get("anchors", [])
    bridges = manifest.get("bridges", [])
    for definitions, identifier, field, row_field in (
        (anchors, "anchor_id", "surface", "anchor_surface"),
        (bridges, "bridge_id", "prefix", "bridge_prefix"),
    ):
        expected = {item["id"]: item for item in definitions}
        if len(expected) != len(definitions):
            raise IncompatibleMeasurement("Manifest coordinate IDs must be unique.")
        if definitions:
            for row in frame[[identifier, row_field]].drop_duplicates().to_dict("records"):
                entry = expected.get(row[identifier])
                if entry is None or entry[field] != row[row_field]:
                    raise IncompatibleMeasurement("Row coordinates disagree with the manifest.")
    if bridges:
        if len({(item.get("relation", "associated_with"), item.get("language", "en"))
                for item in bridges}) != 1:
            raise IncompatibleMeasurement("Bridge aggregation requires one relation and language.")
        by_id = {item["id"]: item for item in bridges}
        for row in frame[["bridge_id", "relation"]].drop_duplicates().to_dict("records"):
            if by_id[row["bridge_id"]].get("relation", "associated_with") != row["relation"]:
                raise IncompatibleMeasurement("Bridge relation disagrees with the manifest.")
    for key in ["event"]:
        if key in manifest and manifest[key] != frame[key].iloc[0]:
            raise IncompatibleMeasurement(f"Manifest {key} differs from score rows.")
    instrument = manifest.get("instrument") or {}
    if "event" in instrument and instrument["event"] != frame.event.iloc[0]:
        raise IncompatibleMeasurement("Instrument event differs from score rows.")
    for model in [manifest.get("model"), instrument.get("model")]:
        if isinstance(model, dict):
            for model_key, row_key in [("id", "model_id"), ("revision", "model_revision")]:
                if model_key in model and model[model_key] != frame[row_key].iloc[0]:
                    raise IncompatibleMeasurement("Instrument model differs from score rows.")
    return {
        "instrument": deepcopy(manifest.get("instrument")),
        **{key: deepcopy(manifest.get(key)) for key in
           ["model", "backend", "resources", "system_prompt"]},
        "event": frame["event"].iloc[0],
        "row_model": frame[["model_id", "model_revision"]].iloc[0].to_dict(),
        "anchors": sorted(deepcopy(anchors), key=lambda item: item["id"])
        if anchors else frame[["anchor_id", "anchor_surface"]].drop_duplicates()
        .sort_values("anchor_id").to_dict("records"),
        "bridges": sorted(deepcopy(bridges), key=lambda item: item["id"])
        if bridges else frame[["bridge_id", "bridge_prefix", "relation"]].drop_duplicates()
        .sort_values("bridge_id").to_dict("records"),
    }


def _grid(scores: ScoreTable, variant: str, missing: str = "error"):
    """Return a fixed sample × anchor × bridge grid and explicit coverage."""
    _configuration(scores)
    if variant not in _VARIANTS or variant not in scores.frame:
        raise MissingScores(f"Unavailable score variant: {variant}")
    if missing not in {"error", "drop_samples", "drop_anchors"}:
        raise ValueError("missing must be error, drop_samples, or drop_anchors.")
    frame = scores.frame
    ids = []
    for field, manifest_key in zip(_KEYS, ["samples", "anchors", "bridges"]):
        definitions = scores.manifest.get(manifest_key, [])
        values = tuple(item["id"] for item in definitions) if definitions else tuple(
            frame[field].drop_duplicates())
        if len(set(values)) != len(values) or not values:
            raise IncompatibleMeasurement(f"Invalid {manifest_key} identities.")
        if not set(frame[field]).issubset(values):
            raise IncompatibleMeasurement(f"Rows contain undeclared {field} values.")
        ids.append(values)
    sample_ids, anchor_ids, bridge_ids = ids
    index = pd.MultiIndex.from_product(ids, names=_KEYS)
    values = pd.to_numeric(frame[variant], errors="coerce").where(frame.status.eq("ok"))
    series = pd.Series(values.to_numpy(), index=pd.MultiIndex.from_frame(frame[_KEYS]))
    cube = series.reindex(index).to_numpy(dtype=np.float64).reshape(*(len(v) for v in ids))
    good = np.isfinite(cube)
    coverage = {"expected_cells": int(cube.size), "valid_cells": int(good.sum()),
                "missing_policy": missing, "dropped_samples": [], "dropped_anchors": [],
                "bridge_ids": list(bridge_ids)}
    if not good.all():
        if missing == "error":
            raise MissingScores(f"Incomplete finite score grid: {good.sum()}/{cube.size} cells.")
        if missing == "drop_samples":
            keep = good.all(axis=(1, 2))
            coverage["dropped_samples"] = [s for s, ok in zip(sample_ids, keep) if not ok]
            sample_ids = tuple(s for s, ok in zip(sample_ids, keep) if ok)
            cube = cube[keep]
        else:
            keep = good.all(axis=(0, 2))
            coverage["dropped_anchors"] = [a for a, ok in zip(anchor_ids, keep) if not ok]
            anchor_ids = tuple(a for a, ok in zip(anchor_ids, keep) if ok)
            cube = cube[:, keep]
        if not cube.size:
            raise MissingScores("The explicit missing-data policy removed every coordinate.")
    coverage["retained_cells"] = int(cube.size)
    return cube, sample_ids, anchor_ids, bridge_ids, coverage


def fit_reference(scores: ScoreTable, *, split: str | None = None) -> CalibrationArtifact:
    """Fit logmeanexp, mean-logp, and sample SD from a complete reference grid.

    Known reference splits must be train/training or external. Explicit split
    arguments cannot override a test/validation split recorded on a sample.
    """
    known = [split, scores.manifest.get("split")]
    for source in [scores.manifest, scores.manifest.get("dataset_manifest", {})]:
        assignments = source.get("splits", {})
        if isinstance(assignments, dict):
            known.extend(assignments.get(sample_id) for sample_id in
                         scores.frame.sample_id.drop_duplicates())
    for sample in scores.manifest.get("samples", []):
        known.extend([sample.get("split"), sample.get("metadata", {}).get("split")])
    known = [str(value).lower() for value in known if value is not None]
    if any(value not in {"train", "training", "external"} for value in known):
        raise IncompatibleMeasurement("References must use training or external data, never test/validation.")
    cube, sample_ids, anchor_ids, bridge_ids, coverage = _grid(scores, "raw_logp")
    rows = []
    for j, anchor_id in enumerate(anchor_ids):
        for b, bridge_id in enumerate(bridge_ids):
            values = cube[:, j, b]
            sd = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            rows.append({"anchor_id": anchor_id, "bridge_id": bridge_id,
                         "reference_n": len(values),
                         "log_mean_probability": float(logsumexp(values) - np.log(len(values))),
                         "mean_logp": float(values.mean()), "std_logp": sd,
                         "z_status": "ok" if np.isfinite(sd) and sd > 0 else
                         ("undefined_constant" if sd == 0 else "undefined_insufficient")})
    return CalibrationArtifact(pd.DataFrame(rows), {
        "configuration": _configuration(scores), "reference_sample_ids": list(sample_ids),
        "reference_splits": sorted(set(known)) or ["unspecified"],
        "reference_measurement_id": scores.manifest.get("measurement_id"),
        "reference_samples": deepcopy(scores.manifest.get("samples", [])),
        "std_ddof": 1, "coverage": coverage,
    })


def transform(scores: ScoreTable, reference: CalibrationArtifact, *,
              variant: str | None = None) -> ScoreTable:
    """Append reference log-ratio and/or reference-z without overwriting raw scores."""
    if variant not in {None, "reference_log_ratio", "reference_z"}:
        raise ValueError("Calibration variant must be reference_log_ratio or reference_z.")
    if _configuration(scores) != reference.manifest.get("configuration"):
        raise IncompatibleMeasurement("Reference instrument, events, or coordinates differ.")
    stats = reference.statistics
    if stats.duplicated(["anchor_id", "bridge_id"]).any():
        raise IncompatibleMeasurement("Reference statistics contain duplicate coordinates.")
    required = {"anchor_id", "bridge_id", "log_mean_probability", "mean_logp", "std_logp"}
    if not required.issubset(stats.columns):
        raise IncompatibleMeasurement("Reference statistics are incomplete.")
    baseline = stats[["log_mean_probability", "mean_logp"]].to_numpy(dtype=float)
    scales = stats.std_logp.to_numpy(dtype=float)
    if not np.isfinite(baseline).all() or np.isinf(scales).any() or np.any(scales < 0):
        raise IncompatibleMeasurement("Reference statistics contain invalid probabilities or scales.")
    frame = scores.frame.copy()
    joined = frame[["anchor_id", "bridge_id"]].merge(stats, how="left", sort=False,
                                                    on=["anchor_id", "bridge_id"], validate="many_to_one")
    if joined["log_mean_probability"].isna().any():
        raise IncompatibleMeasurement("Reference lacks required coordinates.")
    raw = pd.to_numeric(frame.raw_logp, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(raw) & frame.status.eq("ok").to_numpy()
    if variant in {None, "reference_log_ratio"}:
        frame["reference_log_ratio"] = np.where(
            valid, raw - joined.log_mean_probability.to_numpy(), np.nan)
    if variant in {None, "reference_z"}:
        sd = joined.std_logp.to_numpy(dtype=float)
        z_valid = valid & np.isfinite(sd) & (sd > 0)
        z = np.full(len(frame), np.nan)
        np.divide(raw - joined.mean_logp.to_numpy(), sd, out=z, where=z_valid)
        frame["reference_z"] = z
        frame["reference_z_status"] = np.where(z_valid, "ok", "undefined_reference_scale")
        frame.loc[~valid, "reference_z_status"] = "failed_score"
    manifest = deepcopy(scores.manifest)
    manifest["calibration"] = deepcopy(reference.manifest)
    return ScoreTable(frame, manifest)


def to_matrix(scores: ScoreTable, *, variant: str = "raw_logp",
              missing: str = "error") -> MeasurementMatrix:
    """Uniformly average a fixed bridge set; never average over missing bridges."""
    cube, sample_ids, anchor_ids, bridge_ids, coverage = _grid(scores, variant, missing)
    manifest = deepcopy(scores.manifest)
    manifest.update({"aggregation": "uniform_mean_across_bridges", "bridge_ids": list(bridge_ids),
                     "coverage": coverage, "variant": variant})
    return MeasurementMatrix(cube.mean(axis=2), sample_ids, anchor_ids, variant, manifest)
