"""Source-resampled sensitivity, without treating pair edges as independent cases."""
from itertools import combinations

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from vlanchor.evaluation import group_bootstrap


def rank_agreement(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Rank comparisons need aligned finite values.")
    return float(spearmanr(a, b).statistic) if a.size > 1 and np.ptp(a) and np.ptp(b) else np.nan


def interval_or_undefined(statistic, groups):
    point = float(statistic(np.arange(len(groups))))
    if not np.isfinite(point) or len(set(groups)) < 2:
        return {"estimate": point if np.isfinite(point) else None,
            "lower": None, "upper": None, "status": "undefined_statistic_or_insufficient_sources"}
    try:
        return {"status": "ok", **group_bootstrap(statistic, groups)}
    except ValueError as exc:
        if "Too many undefined bootstrap" not in str(exc):
            raise
        return {"estimate": point, "lower": None, "upper": None,
            "status": "insufficient_defined_bootstrap_replicates"}


def pair_distance_agreement(a, b, index=None):
    """Resampling nodes repeats their incident edges, excluding copies of a self edge."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Require aligned square pair-distance matrices.")
    index = np.arange(len(a)) if index is None else np.asarray(index)
    row, col = np.triu_indices(len(index), 1)
    keep = index[row] != index[col]
    return rank_agreement(a[index[row[keep]], index[col[keep]]], b[index[row[keep]], index[col[keep]]])


def geometry_agreement(a, b, groups):
    if len(a) != len(b) or len(a) != len(groups):
        raise ValueError("Both geometries must cover the same source-labelled materials.")
    da, db = cdist(a, a), cdist(b, b)
    return interval_or_undefined(lambda index: pair_distance_agreement(da, db, index), groups)


def template_clusters(cube, groups, *, n_resamples=1000):
    """Fixed k=2, seed42/n_init10; source bootstrap refits then predicts original nodes."""
    cube = np.asarray(cube, dtype=float)
    if cube.ndim != 3 or len(cube) != len(groups) or cube.shape[2] != 3 or not np.isfinite(cube).all():
        raise ValueError("Require complete sample×anchor×three-template scores.")
    representations = [cube.mean(axis=2), *[cube[:, :, j] for j in range(3)]]

    def labels(values, index):
        if len(np.unique(values[index], axis=0)) < 2:
            return None
        return KMeans(n_clusters=2, random_state=42, n_init=10).fit(values[index]).predict(values)

    original = [labels(values, np.arange(len(cube))) for values in representations]
    if any(value is None for value in original) or len(set(groups)) < 2:
        return {"status": "insufficient_distinct_vectors_or_sources", "k": 2}, None
    pairs = list(combinations(range(1, 4), 2))
    point = float(np.mean([adjusted_rand_score(original[i], original[j]) for i, j in pairs]))
    units = np.asarray(groups)
    blocks = [np.flatnonzero(units == unit) for unit in np.unique(units)]
    rng = np.random.default_rng(42)
    template, stability = [], []
    for _ in range(n_resamples):
        index = np.concatenate([blocks[j] for j in rng.integers(len(blocks), size=len(blocks))])
        fitted = [labels(values, index) for values in representations]
        if any(value is None for value in fitted):
            continue
        template.append(float(np.mean([adjusted_rand_score(fitted[i], fitted[j]) for i, j in pairs])))
        stability.append(float(adjusted_rand_score(original[0], fitted[0])))
    adequate = len(template) >= .8*n_resamples
    return {"status": "ok" if adequate else "insufficient_defined_bootstrap_replicates", "k": 2,
        "seed": 42, "n_init": 10, "source_groups": len(blocks), "valid_resamples": len(template),
        "invalid_resamples": n_resamples-len(template), "template_mean_ari": point,
        "template_ari_interval": np.quantile(template, [.025, .975]).tolist() if adequate else None,
        "fixed_node_bootstrap_ari_mean": float(np.mean(stability)) if stability else None,
        "fixed_node_bootstrap_ari_interval": np.quantile(stability, [.025, .975]).tolist() if adequate else None,
        "scope": "Descriptive stability conditional on these materials, not cluster validity or superiority"}, np.stack(original)
