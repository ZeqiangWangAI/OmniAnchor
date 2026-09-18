"""Exact distance reuse for source-resampled DWUG analyses (statistical CPU work)."""
import numpy as np
from scipy.spatial.distance import cdist


def distances_from_indices(values, distances, a, b):
    if not len(a) or not len(b):
        raise ValueError("Both periods must be nonempty.")
    weights = np.bincount(a, minlength=len(values))/len(a) - np.bincount(b, minlength=len(values))/len(b)
    energy = float(-weights @ distances @ weights)
    if energy < -1e-10 * max(1., float(distances.max())):
        raise ArithmeticError("Negative energy beyond floating-point cancellation.")
    return np.array([np.linalg.norm(weights @ values), max(0., energy)])


def source_shift(values, periods, units, *, seed=42, n_bootstrap=1000, n_permutations=1000):
    values = np.asarray(values, dtype=float)
    periods, units = np.asarray(periods), np.asarray(units)
    times = sorted(set(periods))
    if len(times) != 2 or not np.isfinite(values).all() or len(values) != len(units):
        raise ValueError("Require complete finite values and exactly two periods.")
    blocks = {u: np.flatnonzero(units == u) for u in np.unique(units)}
    if any(len(set(periods[b])) != 1 for b in blocks.values()):
        raise ValueError("Source occurs in both periods; this unpaired design does not apply.")
    by_period = [[u for u, b in blocks.items() if periods[b[0]] == t] for t in times]
    distances = cdist(values, values)
    a, b = [np.flatnonzero(periods == t) for t in times]
    point = distances_from_indices(values, distances, a, b)
    rng = np.random.default_rng(seed)
    boot = []
    if min(map(len, by_period)) >= 2:
        for _ in range(n_bootstrap):
            aa, bb = [np.concatenate([blocks[pool[i]] for i in rng.integers(len(pool), size=len(pool))]) for pool in by_period]
            boot.append(distances_from_indices(values, distances, aa, bb))
    intervals = np.quantile(boot, [.025, .975], axis=0) if boot else np.full((2, 2), np.nan)
    # Genre is the published document filename prefix; shuffle whole documents,
    # preserving the observed number of source documents per period within genre.
    strata = {}
    for unit, rows in blocks.items():
        genre = unit.split("_", 1)[0]
        strata.setdefault(genre, []).append((unit, periods[rows[0]]))
    exchangeable = any(len({t for _, t in items}) == 2 for items in strata.values())
    exceed = 0
    performed = n_permutations if exchangeable and boot else 0
    for _ in range(performed):
        aa, bb = [], []
        for items in strata.values():
            shuffled = rng.permutation([t for _, t in items])
            for (unit, _), time in zip(items, shuffled):
                (aa if time == times[0] else bb).extend(blocks[unit])
        null = distances_from_indices(values, distances, np.array(aa), np.array(bb))
        exceed += null[1] >= point[1]-1e-12
    balanced, within = [], []
    for _ in range(100):
        aa, bb = rng.choice(a, min(len(a), len(b)), replace=False), rng.choice(b, min(len(a), len(b)), replace=False)
        balanced.append(distances_from_indices(values, distances, aa, bb))
        for pool in by_period:
            if len(pool) < 2:
                continue
            order = rng.permutation(len(pool))
            half = len(pool)//2
            aa = np.concatenate([blocks[pool[i]] for i in order[:half]])
            bb = np.concatenate([blocks[pool[i]] for i in order[half:]])
            within.append(distances_from_indices(values, distances, aa, bb))
    return {"centroid_distance": point[0], "energy_distance": point[1],
        "centroid_ci_lower": intervals[0, 0], "centroid_ci_upper": intervals[1, 0],
        "energy_ci_lower": intervals[0, 1], "energy_ci_upper": intervals[1, 1],
        "bootstrap_replicates": len(boot), "n_sources_0": len(by_period[0]), "n_sources_1": len(by_period[1]),
        "n_rows_0": len(a), "n_rows_1": len(b), "permutations_run": performed,
        "genre_stratified_permutation_p": (1+exceed)/(1+performed) if performed else np.nan,
        "balanced_centroid_mean": np.mean(balanced, axis=0)[0], "balanced_energy_mean": np.mean(balanced, axis=0)[1],
        "within_period_energy_mean": np.mean(within, axis=0)[1] if within else np.nan,
        "uncertainty_status": "conditional_source_bootstrap" if boot else "undefined_insufficient_sources",
        "permutation_assumption": "whole source documents exchangeable within genre under no period effect; temporal stylistic/corpus changes may violate this assumption"}
