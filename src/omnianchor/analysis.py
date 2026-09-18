"""Descriptive analyses on a fixed, explicitly identified measurement matrix."""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .errors import MissingScores
from .types import MeasurementMatrix


def _values(matrix: MeasurementMatrix) -> np.ndarray:
    values = np.asarray(matrix.values, dtype=np.float64)
    if not values.size or not np.isfinite(values).all():
        raise MissingScores("Analysis requires a non-empty, complete finite matrix.")
    return values


def _iterations(n: int, name: str) -> None:
    if not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")


def cluster(matrix: MeasurementMatrix, *, k: int, seed: int = 42) -> dict[str, Any]:
    values = _values(matrix)
    if not isinstance(k, (int, np.integer)) or not 1 <= k <= len(values):
        raise ValueError("k must be between one and the number of samples.")
    if len(np.unique(values, axis=0)) < k:
        raise ValueError("k exceeds the number of distinct sample vectors.")
    estimator = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(values)
    return {"sample_ids": list(matrix.sample_ids), "labels": estimator.labels_.tolist(),
            "centroids": estimator.cluster_centers_.tolist(), "inertia": float(estimator.inertia_),
            "k": k, "seed": seed, "variant": matrix.variant}


def pca(matrix: MeasurementMatrix, *, n_components: int = 2) -> dict[str, Any]:
    values = _values(matrix)
    if not isinstance(n_components, (int, np.integer)) or not 1 <= n_components <= min(values.shape):
        raise ValueError("n_components must not exceed samples or anchors.")
    columns = [f"PC{i + 1}" for i in range(n_components)]
    if len(values) < 2 or not np.any(values.std(axis=0) > 0):
        coordinates = np.zeros((len(values), n_components))
        ratios = [np.nan] * n_components
        status = "undefined_variance"
    else:
        estimator = PCA(n_components=n_components, svd_solver="full")
        coordinates = estimator.fit_transform(values)
        ratios = estimator.explained_variance_ratio_.tolist()
        status = "ok"
    frame = pd.DataFrame(coordinates, columns=columns)
    frame.insert(0, "sample_id", matrix.sample_ids)
    return {"coordinates": frame, "explained_variance_ratio": ratios, "status": status,
            "variant": matrix.variant}


def _metadata(matrix: MeasurementMatrix, metadata=None) -> list[dict[str, Any]]:
    source = matrix.manifest.get("samples", []) if metadata is None else metadata
    if isinstance(source, pd.DataFrame):
        source = source.to_dict("records")
    if isinstance(source, Mapping):
        lookup = {key: dict(value) for key, value in source.items()}
    else:
        lookup = {item.get("sample_id", item.get("id")): item for item in source}
        if len(lookup) != len(source):
            raise ValueError("Metadata contains duplicate sample identities.")
    return [dict(lookup.get(sample_id, {})) for sample_id in matrix.sample_ids]


def _labels(matrix: MeasurementMatrix, labels, *, field: str) -> list[Any]:
    if labels is None:
        result = [item.get(field, item.get("metadata", {}).get(field)) for item in _metadata(matrix)]
    elif isinstance(labels, Mapping):
        result = [labels.get(sample_id) for sample_id in matrix.sample_ids]
    else:
        result = list(labels)
    if len(result) != len(matrix.sample_ids) or any(pd.isna(value) for value in result):
        raise ValueError(f"Every sample requires one {field} label.")
    return result


def _interval(values: np.ndarray) -> tuple[Any, Any]:
    if not len(values):
        return np.nan, np.nan
    lower, upper = np.quantile(values, [0.025, 0.975], axis=0)
    return lower, upper


def _sampling_units(matrix: MeasurementMatrix, sampling_units=None, metadata=None):
    if sampling_units is not None:
        units = _labels(matrix, sampling_units, field="sampling_unit")
        source = "explicit_sampling_units"
    else:
        info = _metadata(matrix, metadata)
        units = []
        for sample_id, item in zip(matrix.sample_ids, info):
            group_id = item.get("group_id")
            if group_id is None or pd.isna(group_id):
                group_id = item.get("metadata", {}).get("group_id")
            # Namespaces prevent a fallback sample ID colliding with a source group ID.
            units.append(("sample", sample_id) if group_id is None or pd.isna(group_id)
                         else ("group", group_id))
        source = "manifest_group_id_or_sample_id" if metadata is None else "metadata_group_id_or_sample_id"
    try:
        set(units)
    except TypeError as exc:
        raise ValueError("Sampling units must be scalar, hashable identities.") from exc
    return units, source


def _cluster_plan(mask_a: np.ndarray, mask_b: np.ndarray, units: list):
    def membership(mask):
        groups = {}
        for i in np.flatnonzero(mask):
            groups.setdefault(units[i], []).append(i)
        return groups

    a, b = membership(mask_a), membership(mask_b)
    shared = set(a) & set(b)
    if shared and set(a) != set(b):
        raise ValueError("Sampling units partially overlap across cohorts. Use fully paired or "
                         "disjoint source groups; mixed overlap is not automatically modeled.")
    paired = bool(shared)
    order_b = list(a) if paired else list(b)
    return {"a": [np.asarray(a[key]) for key in a],
            "b": [np.asarray(b[key]) for key in order_b],
            "bootstrap_design": "paired_clusters" if paired else "independent_clusters",
            "n_units_a": len(a), "n_units_b": len(b), "n_shared_units": len(shared)}


def _draw_clusters(plan: dict, rng: np.random.Generator):
    indices_a = rng.integers(len(plan["a"]), size=len(plan["a"]))
    indices_b = indices_a if plan["bootstrap_design"] == "paired_clusters" else rng.integers(
        len(plan["b"]), size=len(plan["b"]))
    return (np.concatenate([plan["a"][i] for i in indices_a]),
            np.concatenate([plan["b"][i] for i in indices_b]))


def _permute_clusters(plan: dict, rng: np.random.Generator):
    if plan["bootstrap_design"] == "paired_clusters":
        swap = rng.integers(2, size=len(plan["a"]))
        left = [plan["b"][i] if flip else plan["a"][i] for i, flip in enumerate(swap)]
        right = [plan["a"][i] if flip else plan["b"][i] for i, flip in enumerate(swap)]
    else:
        groups = plan["a"] + plan["b"]
        order = rng.permutation(len(groups))
        left = [groups[i] for i in order[:len(plan["a"])]]
        right = [groups[i] for i in order[len(plan["a"]):]]
    return np.concatenate(left), np.concatenate(right)


def compare_groups(matrix: MeasurementMatrix, groups=None, *, group_a=None, group_b=None,
                   sampling_units=None, n_bootstrap: int = 1000, seed: int = 42) -> dict[str, Any]:
    """Cluster bootstrap of row-weighted means; shared source groups stay paired."""
    values = _values(matrix)
    _iterations(n_bootstrap, "n_bootstrap")
    labels = _labels(matrix, groups, field="group_id")
    unique = list(dict.fromkeys(labels))
    if group_a is None and group_b is None and len(unique) == 2:
        group_a, group_b = unique
    if group_a is None or group_b is None or group_a == group_b:
        raise ValueError("Select two distinct groups (required when more than two exist).")
    mask_a = np.array([label == group_a for label in labels])
    mask_b = np.array([label == group_b for label in labels])
    a, b = values[mask_a], values[mask_b]
    if not len(a) or not len(b):
        raise ValueError("Both selected groups must contain samples.")
    delta = b.mean(axis=0) - a.mean(axis=0)
    units, unit_source = _sampling_units(matrix, sampling_units)
    plan = _cluster_plan(mask_a, mask_b, units)
    enough_units = min(plan["n_units_a"], plan["n_units_b"]) >= 2
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_bootstrap if enough_units else 0):
        indices_a, indices_b = _draw_clusters(plan, rng)
        draws.append(values[indices_b].mean(axis=0) - values[indices_a].mean(axis=0))
    draws = np.asarray(draws)
    lo, hi = _interval(draws)
    distance_lo, distance_hi = _interval(np.linalg.norm(draws, axis=1)) if len(draws) else (np.nan, np.nan)
    differences = pd.DataFrame({"anchor_id": matrix.anchor_ids, "mean_a": a.mean(axis=0),
                                "mean_b": b.mean(axis=0), "difference": delta,
                                "ci_lower": lo, "ci_upper": hi})
    return {"group_a": group_a, "group_b": group_b, "n_a": len(a), "n_b": len(b),
            "differences": differences, "centroid_distance": float(np.linalg.norm(delta)),
            "distance_ci_lower": float(distance_lo), "distance_ci_upper": float(distance_hi),
            "n_bootstrap": n_bootstrap, "bootstrap_replicates": len(draws), "seed": seed,
            "bootstrap_unit": "source_group", "sampling_unit_source": unit_source,
            "bootstrap_design": plan["bootstrap_design"], "n_units_a": plan["n_units_a"],
            "n_units_b": plan["n_units_b"], "n_shared_units": plan["n_shared_units"],
            "estimand": "row_weighted_group_means",
            "uncertainty_status": "ok" if enough_units and n_bootstrap else
            ("undefined_insufficient_sampling_units" if not enough_units else "not_requested")}


def _distance_mean(a: np.ndarray, b: np.ndarray, block_size: int) -> float:
    total = 0.0
    for i in range(0, len(a), block_size):
        for j in range(0, len(b), block_size):
            total += cdist(a[i:i + block_size], b[j:j + block_size]).sum(dtype=np.float64)
    return float(total / (len(a) * len(b)))


def energy_distance(x: np.ndarray, y: np.ndarray, *, block_size: int = 256) -> float:
    """Exact empirical V-statistic, including diagonal pairs, in bounded blocks."""
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or not a.size or not b.size:
        raise ValueError("Energy distance needs non-empty matrices with the same columns.")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise MissingScores("Energy distance requires finite observations.")
    if not isinstance(block_size, (int, np.integer)) or block_size < 1:
        raise ValueError("block_size must be a positive integer.")
    cross = _distance_mean(a, b, block_size)
    result = 2 * cross - _distance_mean(a, a, block_size) - _distance_mean(b, b, block_size)
    if not np.isfinite(result):
        raise ArithmeticError("Energy distance overflowed; explicitly rescale the coordinates.")
    # The V-statistic is nonnegative; only numerical cancellation is clipped.
    if result < -1e-10 * max(1.0, cross):
        raise ArithmeticError("Unexpected negative energy distance beyond roundoff.")
    return float(max(0.0, result))


def semantic_shift(matrix: MeasurementMatrix, metadata=None, *, target_ids=None, periods=None,
                   sampling_units=None,
                   n_bootstrap: int = 1000, n_permutations: int = 0,
                   seed: int = 42, block_size: int = 256) -> dict[str, Any]:
    """Compare exactly two periods per target; keep undefined targets in the output."""
    values = _values(matrix)
    _iterations(n_bootstrap, "n_bootstrap")
    _iterations(n_permutations, "n_permutations")
    info = _metadata(matrix, metadata)
    if target_ids is None:
        target_ids = []
        for item in info:
            target = item.get("target")
            target_ids.append(item.get("target_id") or item.get("lemma") or
                              (target.get("lemma") if isinstance(target, dict) else target) or
                              item.get("metadata", {}).get("target_id"))
    if periods is None:
        periods = [item.get("time") or item.get("period") or
                   item.get("metadata", {}).get("time") or item.get("metadata", {}).get("period")
                   for item in info]
    targets = _labels(matrix, target_ids, field="target_id")
    times = _labels(matrix, periods, field="time")
    units, unit_source = _sampling_units(matrix, sampling_units, metadata)
    rng = np.random.default_rng(seed)
    rows = []
    for target in dict.fromkeys(targets):
        selected = np.array([value == target for value in targets])
        target_times = list(dict.fromkeys(t for t, keep in zip(times, selected) if keep))
        if len(target_times) != 2:
            rows.append({"target_id": target, "status": "requires_two_periods",
                         "centroid_distance": np.nan, "energy_distance": np.nan,
                         "permutation_pvalue": np.nan})
            continue
        t0, t1 = target_times
        mask_a = selected & np.array([t == t0 for t in times])
        mask_b = selected & np.array([t == t1 for t in times])
        a, b = values[mask_a], values[mask_b]
        plan = _cluster_plan(mask_a, mask_b, units)
        enough_units = min(plan["n_units_a"], plan["n_units_b"]) >= 2
        centroid = float(np.linalg.norm(b.mean(axis=0) - a.mean(axis=0)))
        energy = energy_distance(a, b, block_size=block_size)
        boot = []
        if enough_units:
            for _ in range(n_bootstrap):
                indices_a, indices_b = _draw_clusters(plan, rng)
                aa, bb = values[indices_a], values[indices_b]
                boot.append([np.linalg.norm(bb.mean(axis=0) - aa.mean(axis=0)),
                             energy_distance(aa, bb, block_size=block_size)])
        lo, hi = _interval(np.asarray(boot)) if boot else ([np.nan] * 2, [np.nan] * 2)
        exceed = 0
        permutations_run = n_permutations if enough_units else 0
        for _ in range(permutations_run):
            indices_a, indices_b = _permute_clusters(plan, rng)
            null = energy_distance(values[indices_a], values[indices_b], block_size=block_size)
            exceed += null >= energy - 1e-12
        rows.append({"target_id": target, "period_0": t0, "period_1": t1,
                     "n_0": len(a), "n_1": len(b), "status": "ok",
                     "n_units_0": plan["n_units_a"], "n_units_1": plan["n_units_b"],
                     "n_shared_units": plan["n_shared_units"], "bootstrap_design": plan["bootstrap_design"],
                     "bootstrap_replicates": len(boot), "permutations_run": permutations_run,
                     "uncertainty_status": "ok" if enough_units and n_bootstrap else
                     ("undefined_insufficient_sampling_units" if not enough_units else "not_requested"),
                     "centroid_distance": centroid, "energy_distance": energy,
                     "centroid_ci_lower": lo[0], "centroid_ci_upper": hi[0],
                     "energy_ci_lower": lo[1], "energy_ci_upper": hi[1],
                     "permutation_pvalue": (1 + exceed) / (1 + permutations_run)
                     if permutations_run else np.nan})
    frame = pd.DataFrame(rows)
    frame["permutation_qvalue"] = bh_fdr(frame.permutation_pvalue.to_numpy())
    return {"results": frame, "n_bootstrap": n_bootstrap, "n_permutations": n_permutations,
            "seed": seed, "bootstrap_unit": "source_group", "sampling_unit_source": unit_source,
            "energy_estimator": "empirical_V", "estimand": "row_weighted_period_distributions",
            "permutation_assumption": "whole source groups are exchangeable across periods; "
            "paired groups use within-group period-block swaps",
            "variant": matrix.variant}


def concept_network(matrix: MeasurementMatrix) -> dict[str, Any]:
    """Full signed Pearson coactivation network; undefined edges remain explicit."""
    values = _values(matrix)
    centered = values - values.mean(axis=0)
    norms = np.linalg.norm(centered, axis=0)
    valid = (norms > 0) & (len(values) >= 2)
    normalized = np.zeros_like(centered)
    np.divide(centered, norms, out=normalized, where=valid[None, :])
    correlations = np.clip(normalized.T @ normalized, -1, 1)
    correlations[~valid, :] = np.nan
    correlations[:, ~valid] = np.nan
    nodes = pd.DataFrame({"anchor_id": matrix.anchor_ids,
                          "status": ["ok" if ok else "undefined_variance" for ok in valid]})
    rows = [{"source": matrix.anchor_ids[j], "target": matrix.anchor_ids[k],
             "weight": correlations[j, k],
             "status": "ok" if valid[j] and valid[k] else "undefined_variance"}
            for j in range(values.shape[1]) for k in range(j + 1, values.shape[1])]
    return {"correlation": pd.DataFrame(correlations, index=matrix.anchor_ids,
                                         columns=matrix.anchor_ids),
            "nodes": nodes, "edges": pd.DataFrame(rows, columns=["source", "target", "weight", "status"]),
            "n_samples": len(values), "definition": "signed_Pearson_coactivation"}


def sample_distance_graph(matrix: MeasurementMatrix, *, k: int = 5) -> dict[str, Any]:
    values = _values(matrix)
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer.")
    distances = cdist(values, values)
    effective_k = min(k, len(values) - 1)
    pairs = set()
    for i in range(len(values)):
        neighbors = sorted((j for j in range(len(values)) if j != i),
                           key=lambda j: (distances[i, j], str(matrix.sample_ids[j])))[:effective_k]
        pairs.update(tuple(sorted((i, j))) for j in neighbors)
    edges = pd.DataFrame([{"source": matrix.sample_ids[i], "target": matrix.sample_ids[j],
                           "distance": distances[i, j]} for i, j in sorted(pairs)],
                         columns=["source", "target", "distance"])
    return {"nodes": pd.DataFrame({"sample_id": matrix.sample_ids}), "edges": edges,
            "distances": pd.DataFrame(distances, index=matrix.sample_ids, columns=matrix.sample_ids),
            "k": effective_k, "requested_k": k, "definition": "union_kNN_Euclidean"}


def semantic_network(matrix: MeasurementMatrix, *, kind: str = "concept", **kwargs):
    if kind == "concept":
        return concept_network(matrix, **kwargs)
    if kind in {"sample", "distance"}:
        return sample_distance_graph(matrix, **kwargs)
    raise ValueError("kind must be concept or sample.")


def bh_fdr(pvalues: Sequence[float] | np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg adjusted p-values; NaNs remain undefined."""
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or np.isinf(p).any() or np.any((p < 0) | (p > 1)):
        raise ValueError("pvalues must be one-dimensional values in [0,1], or NaN.")
    result = np.full(p.shape, np.nan)
    indices = np.flatnonzero(np.isfinite(p))
    if len(indices):
        order = indices[np.argsort(p[indices], kind="stable")]
        adjusted = p[order] * len(order) / np.arange(1, len(order) + 1)
        result[order] = np.minimum(1.0, np.minimum.accumulate(adjusted[::-1])[::-1])
    return result


def independent_column_null(matrix: MeasurementMatrix, *, seed: int = 42) -> MeasurementMatrix:
    values = _values(matrix)
    rng = np.random.default_rng(seed)
    shuffled = np.column_stack([rng.permutation(values[:, j]) for j in range(values.shape[1])])
    manifest = deepcopy(matrix.manifest)
    manifest["null_control"] = {"method": "independent_column_permutation", "seed": seed}
    return MeasurementMatrix(shuffled, matrix.sample_ids, matrix.anchor_ids, matrix.variant, manifest)
