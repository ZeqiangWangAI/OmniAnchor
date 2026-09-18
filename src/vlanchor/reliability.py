"""Repeated-template diagnostics for a fixed anchor and semantic relation."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .calibration import _grid
from .types import MeasurementMatrix, ScoreTable


def cronbach_alpha(items: np.ndarray) -> dict[str, Any]:
    """Rows are samples; columns are equivalent templates for ONE anchor.

    Alpha is a reliability diagnostic, not a validity objective. Arbitrary
    anchor dimensions are different constructs and must not be passed here.
    """
    if isinstance(items, MeasurementMatrix):
        raise ValueError("Do not compute alpha across anchors; use repeated templates for one anchor.")
    values = np.asarray(items, dtype=float)
    if values.ndim != 2:
        raise ValueError("Alpha requires a sample-by-template matrix.")
    n, k = values.shape
    result = {"alpha": np.nan, "n_samples": n, "n_templates": k}
    if n < 2 or k < 2:
        return {**result, "status": "undefined_insufficient"}
    if not np.isfinite(values).all():
        return {**result, "status": "undefined_nonfinite"}
    total_variance = values.sum(axis=1).var(ddof=1)
    if total_variance <= 0:
        return {**result, "status": "undefined_total_variance"}
    alpha = k / (k - 1) * (1 - values.var(axis=0, ddof=1).sum() / total_variance)
    return {**result, "alpha": float(alpha), "status": "ok"}


def bridge_reliability(scores: ScoreTable, *, variant: str = "raw_logp") -> dict[str, Any]:
    cube, sample_ids, anchor_ids, bridge_ids, coverage = _grid(scores, variant)
    pair_rows, anchor_rows = [], []
    for j, anchor in enumerate(anchor_ids):
        coefficients = []
        for a, b in combinations(range(len(bridge_ids)), 2):
            x, y = cube[:, j, a], cube[:, j, b]
            valid = len(x) >= 2 and np.ptp(x) > 0 and np.ptp(y) > 0
            coefficient = float(spearmanr(x, y).statistic) if valid else np.nan
            if valid:
                coefficients.append(coefficient)
            pair_rows.append({"anchor_id": anchor, "bridge_a": bridge_ids[a],
                              "bridge_b": bridge_ids[b], "spearman": coefficient,
                              "n_samples": len(sample_ids),
                              "status": "ok" if valid else "undefined_variance_or_sample_size"})
        alpha = cronbach_alpha(cube[:, j, :])
        anchor_rows.append({"anchor_id": anchor,
                            "mean_pair_spearman": float(np.mean(coefficients)) if coefficients else np.nan,
                            "valid_pairs": len(coefficients), "total_pairs": len(bridge_ids) * (len(bridge_ids) - 1) // 2,
                            "alpha": alpha["alpha"], "alpha_status": alpha["status"]})
    return {"pairs": pd.DataFrame(pair_rows, columns=["anchor_id", "bridge_a", "bridge_b",
                                                      "spearman", "n_samples", "status"]),
            "anchors": pd.DataFrame(anchor_rows), "bridge_ids": list(bridge_ids),
            "coverage": coverage, "variant": variant,
            "interpretation": "within-anchor same-relation template consistency, not semantic validity"}


audit_reliability = bridge_reliability


def template_radii(scores: ScoreTable, *, variant: str = "raw_logp") -> dict[str, Any]:
    """Deterministic perturbation bounds over the observed fixed bridge set only."""
    cube, sample_ids, anchor_ids, bridge_ids, coverage = _grid(scores, variant)
    mean = cube.mean(axis=2)
    sample_radii = np.linalg.norm(cube - mean[:, :, None], axis=1).max(axis=1)
    centered = cube - cube.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=0)
    unit = np.zeros_like(centered)
    np.divide(centered, norms[None, :, :], out=unit, where=norms[None, :, :] > 0)
    mean_centered = mean - mean.mean(axis=0)
    mean_norms = np.linalg.norm(mean_centered, axis=0)
    mean_unit = np.zeros_like(mean_centered)
    np.divide(mean_centered, mean_norms[None, :], out=mean_unit, where=mean_norms[None, :] > 0)
    valid = (mean_norms > 0) & (norms > 0).all(axis=1)
    radii = np.linalg.norm(unit - mean_unit[:, :, None], axis=0).max(axis=1)
    radii[~valid] = np.nan
    correlations = np.clip(mean_unit.T @ mean_unit, -1, 1)
    rows = []
    for j, k in combinations(range(len(anchor_ids)), 2):
        good = valid[j] and valid[k]
        bound = radii[j] + radii[k]
        correlation = float(correlations[j, k]) if good else np.nan
        rows.append({"source": anchor_ids[j], "target": anchor_ids[k],
                     "ensemble_correlation": correlation, "error_bound": bound,
                     "lower": max(-1.0, correlation - bound) if good else np.nan,
                     "upper": min(1.0, correlation + bound) if good else np.nan,
                     "status": "ok" if good else "undefined_variance"})
    return {"samples": pd.DataFrame({"sample_id": sample_ids, "radius": sample_radii}),
            "anchors": pd.DataFrame({"anchor_id": anchor_ids, "normalized_radius": radii,
                                      "status": ["ok" if v else "undefined_variance" for v in valid]}),
            "correlation_bounds": pd.DataFrame(rows, columns=["source", "target", "ensemble_correlation",
                                                               "error_bound", "lower", "upper", "status"]),
            "bridge_ids": list(bridge_ids), "coverage": coverage,
            "scope": "observed same-relation templates only; not confidence intervals or validity guarantees",
            "sample_distance_bound": "abs(distance_b(i,j)-distance_mean(i,j)) <= radius_i + radius_j"}
