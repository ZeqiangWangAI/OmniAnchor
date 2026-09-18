"""Evaluation with explicit axes, held-out model selection, and grouped uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, f1_score, mean_squared_error
from sklearn.preprocessing import StandardScaler


def _paired_arrays(labels, scores, *, binary: bool = False) -> tuple[np.ndarray, np.ndarray]:
    y, s = np.asarray(labels, dtype=float), np.asarray(scores, dtype=float)
    if y.shape != s.shape or y.size == 0:
        raise ValueError("Labels/scores must have the same nonempty shape.")
    if not np.isfinite(y).all() or not np.isfinite(s).all():
        raise ValueError("Labels/scores must be finite; missing observations need an explicit mask.")
    if binary and not np.isin(y, [0, 1]).all():
        raise ValueError("Relevance labels must be binary.")
    return y, s


def evaluate_multilabel(labels, scores, anchor_ids: Sequence[str] | None = None) -> dict:
    """Macro AP ranks samples within each anchor; it is not within-sample candidate AP.

    Columns with no positive examples are explicitly excluded from macro AP.
    Scores can be logits or unnormalized associations; no softmax is applied.
    """
    y, s = _paired_arrays(labels, scores, binary=True)
    if y.ndim != 2:
        raise ValueError("Multilabel evaluation expects [samples, anchors].")
    ids = list(anchor_ids) if anchor_ids is not None else [str(i) for i in range(y.shape[1])]
    if len(ids) != y.shape[1] or len(set(ids)) != len(ids):
        raise ValueError("anchor_ids must uniquely identify every column.")
    eligible = y.sum(axis=0) > 0
    per = {name: float(average_precision_score(y[:, j], s[:, j]))
           for j, name in enumerate(ids) if eligible[j]}
    return {"macro_ap": float(np.mean(list(per.values()))) if per else None,
            "micro_ap": float(average_precision_score(y.ravel(), s.ravel())) if y.sum() else None,
            "per_anchor_ap": per, "excluded_anchors": [ids[j] for j in range(len(ids)) if not eligible[j]],
            "sample_count": y.shape[0], "axis": "rank_samples_within_anchor"}


def evaluate_candidates(
    labels: Mapping[str, Sequence[int]], scores: Mapping[str, Sequence[float]], *,
    require_contrast: bool = True,
) -> dict:
    """Mean per-instance AP/MRR over each instance's own ordered candidate list."""
    if set(labels) != set(scores) or not labels:
        raise ValueError("Candidate labels and scores must cover the same nonempty sample ID set.")
    per_ap, per_rr, excluded = {}, {}, []
    for identifier in sorted(labels):
        y, s = _paired_arrays(labels[identifier], scores[identifier], binary=True)
        if y.ndim != 1:
            raise ValueError("Each candidate list must be one-dimensional.")
        if y.sum() == 0 or (require_contrast and y.sum() == len(y)):
            excluded.append(identifier)
            continue
        order = np.argsort(-s, kind="stable")
        per_ap[identifier] = float(average_precision_score(y, s))
        per_rr[identifier] = float(1 / (np.flatnonzero(y[order])[0] + 1))
    return {"mean_instance_ap": float(np.mean(list(per_ap.values()))) if per_ap else None,
            "mrr": float(np.mean(list(per_rr.values()))) if per_rr else None,
            "per_instance_ap": per_ap, "excluded_samples": excluded,
            "axis": "rank_candidates_within_sample", "mrr_tie_policy": "stable_input_order"}


def evaluate_vad(labels, scores, dimensions: Sequence[str] = ("V", "A", "D")) -> dict:
    y, s = _paired_arrays(labels, scores)
    if y.ndim != 2 or y.shape[1] != len(dimensions):
        raise ValueError("VAD expects [samples, dimensions] with named columns.")
    correlations, excluded = {}, []
    for j, dimension in enumerate(dimensions):
        if len(y) < 2 or np.ptp(y[:, j]) == 0 or np.ptp(s[:, j]) == 0:
            excluded.append(dimension)
        else:
            correlations[dimension] = float(spearmanr(y[:, j], s[:, j]).statistic)
    return {"spearman": correlations, "mean_spearman": float(np.mean(list(correlations.values())))
            if correlations else None, "undefined_dimensions": excluded, "sample_count": len(y)}


def retrieval_metrics(relevance, scores, ks: Sequence[int] = (1, 5, 10)) -> dict:
    """Report both success@K and fraction-of-relevant-items recall@K explicitly."""
    y, s = _paired_arrays(relevance, scores, binary=True)
    if y.ndim != 2 or not ks or any(k < 1 for k in ks):
        raise ValueError("Retrieval expects [queries, candidates] and positive K values.")
    eligible = y.sum(axis=1) > 0
    order = np.argsort(-s, axis=1, kind="stable")
    metrics = {}
    for k in ks:
        retrieved = np.take_along_axis(y, order[:, :min(k, y.shape[1])], axis=1).sum(axis=1)
        metrics[f"success@{k}"] = float(np.mean(retrieved[eligible] > 0)) if eligible.any() else None
        metrics[f"recall@{k}"] = float(np.mean(retrieved[eligible] / y.sum(axis=1)[eligible])) if eligible.any() else None
    return {**metrics, "excluded_queries": np.flatnonzero(~eligible).tolist(),
            "tie_policy": "stable_candidate_order", "query_count": len(y)}


def group_bootstrap(
    statistic: Callable[[np.ndarray], float], groups: Sequence[str], *,
    n_resamples: int = 1000, confidence: float = 0.95, seed: int = 42,
    min_valid_fraction: float = 0.8,
) -> dict:
    """Resample whole groups with replacement; statistic receives repeated row indices.

    Use the same callback on paired method differences for paired confidence intervals.
    Nonfinite replicates are counted; excessive undefined replicates raise an error.
    """
    groups = np.asarray(groups, dtype=str)
    if groups.ndim != 1 or not len(groups) or n_resamples < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap inputs.")
    if not 0 < min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction must be in (0, 1].")
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("A group bootstrap requires at least two independent groups.")
    blocks = [np.flatnonzero(groups == group) for group in unique]
    point = float(statistic(np.arange(len(groups))))
    if not np.isfinite(point):
        raise ValueError("Point statistic is undefined.")
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_resamples):
        index = np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))])
        value = float(statistic(index))
        if np.isfinite(value):
            draws.append(value)
    if len(draws) < n_resamples * min_valid_fraction:
        raise ValueError("Too many undefined bootstrap replicates; use an identifiable statistic.")
    alpha = (1 - confidence) / 2
    lower, upper = np.quantile(draws, [alpha, 1 - alpha])
    return {"estimate": point, "lower": float(lower), "upper": float(upper),
            "confidence": confidence, "independent_groups": len(unique),
            "valid_resamples": len(draws), "invalid_resamples": n_resamples - len(draws), "seed": seed}


def compare_networks(predicted, reference, *, top_k: int = 20) -> dict:
    """Compare undirected adjacency weights on matched, identically ordered nodes."""
    gold, pred = _paired_arrays(reference, predicted)
    if gold.ndim != 2 or gold.shape[0] != gold.shape[1] or len(gold) < 2:
        raise ValueError("Networks must be square adjacency matrices with at least two nodes.")
    if not np.allclose(gold, gold.T) or not np.allclose(pred, pred.T) or top_k < 1:
        raise ValueError("Networks must be symmetric and top_k positive.")
    upper = np.triu_indices(len(gold), k=1)
    g, p = gold[upper], pred[upper]
    rho = float(spearmanr(g, p).statistic) if len(g) > 1 and np.ptp(g) and np.ptp(p) else None
    # Zero-weight entries are not edges and must not pad a top-K graph.
    gi = [int(i) for i in np.argsort(-np.abs(g), kind="stable") if g[i] != 0][:top_k]
    pi = [int(i) for i in np.argsort(-np.abs(p), kind="stable") if p[i] != 0][:top_k]
    common = set(gi) & set(pi)
    return {"edge_weight_spearman": rho,
            "top_k_edge_jaccard": len(common) / len(set(gi) | set(pi)) if gi or pi else None,
            "common_edge_sign_agreement": float(np.mean([np.sign(g[i]) == np.sign(p[i]) for i in common]))
            if common else None, "top_k": top_k, "compared_edge_count": len(g),
            "node_alignment": "caller_supplied_identical_order"}


@dataclass
class LinearProbeResult:
    task: str
    scaler: StandardScaler
    estimator: object
    best_parameter: float
    dev_score: float
    trials: list[dict]

    def predict_scores(self, features) -> np.ndarray:
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError("Probe features must be a finite matrix.")
        x = self.scaler.transform(x)
        if self.task == "multilabel":
            return np.column_stack([np.full(len(x), model) if isinstance(model, float)
                                    else model.predict_proba(x)[:, 1] for model in self.estimator])
        return np.asarray(self.estimator.predict(x))


def fit_linear_probe(
    train_features, train_labels, dev_features, dev_labels, *, task: str = "multilabel",
    grid: Sequence[float] | None = None, seed: int = 42,
) -> LinearProbeResult:
    """Fit scaler/models only on train; select one hyperparameter on dev, never refit on dev.

    Multilabel: independent L2 logistic models, C grid [.01,.1,1,10], macro AP.
    Regression: multi-output ridge, alpha grid [.01,.1,1,10,100], negative MSE.
    Multiclass: L2 logistic, same C grid, macro F1. Ties choose the first grid entry.
    """
    if task not in {"multilabel", "regression", "multiclass"}:
        raise ValueError("Probe task must be multilabel, regression, or multiclass.")
    xt, xd = np.asarray(train_features, dtype=float), np.asarray(dev_features, dtype=float)
    yt, yd = np.asarray(train_labels), np.asarray(dev_labels)
    if xt.ndim != 2 or xd.ndim != 2 or xt.shape[1] != xd.shape[1] or not len(xt) or not len(xd):
        raise ValueError("Train/dev features need matching columns and nonempty rows.")
    if len(xt) != len(yt) or len(xd) != len(yd) or not np.isfinite(xt).all() or not np.isfinite(xd).all():
        raise ValueError("Invalid train/dev alignment or nonfinite features.")
    if task != "multiclass":
        yt, yd = np.asarray(yt, dtype=float), np.asarray(yd, dtype=float)
        if yt.ndim != 2 or yd.ndim != 2 or yt.shape[1] != yd.shape[1]:
            raise ValueError("Multilabel/regression labels require matching two-dimensional columns.")
        if not np.isfinite(yt).all() or not np.isfinite(yd).all():
            raise ValueError("Probe labels must be finite.")
    if task == "multilabel" and (not np.isin(yt, [0, 1]).all() or not np.isin(yd, [0, 1]).all()):
        raise ValueError("Multilabel targets must be binary.")
    values = list(grid) if grid is not None else ([0.01, 0.1, 1.0, 10.0, 100.0] if task == "regression"
                                                 else [0.01, 0.1, 1.0, 10.0])
    if not values or any(not np.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Regularization grid must contain positive finite numbers.")
    scaler = StandardScaler().fit(xt)
    xt_scaled = scaler.transform(xt)
    best, trials = None, []
    for value in values:
        if task == "multilabel":
            models = []
            for j in range(yt.shape[1]):
                if np.unique(yt[:, j]).size == 1:
                    models.append(float(yt[0, j]))
                else:
                    models.append(LogisticRegression(C=value, solver="lbfgs", max_iter=2000,
                                                     random_state=seed).fit(xt_scaled, yt[:, j]))
            estimator = models
        elif task == "regression":
            estimator = Ridge(alpha=value).fit(xt_scaled, yt)
        else:
            if yt.ndim != 1 or yd.ndim != 1 or len(np.unique(yt)) < 2:
                raise ValueError("Multiclass labels need one dimension and at least two train classes.")
            estimator = LogisticRegression(C=value, max_iter=2000, random_state=seed).fit(xt_scaled, yt)
        result = LinearProbeResult(task, scaler, estimator, float(value), float("nan"), [])
        predictions = result.predict_scores(xd)
        score = (evaluate_multilabel(yd, predictions)["macro_ap"] if task == "multilabel" else
                 -mean_squared_error(yd, predictions) if task == "regression" else
                 f1_score(yd, predictions, average="macro", zero_division=0))
        if score is None or not np.isfinite(score):
            raise ValueError("Development selection metric is undefined.")
        result.dev_score = float(score)
        trials.append({"parameter": float(value), "dev_score": float(score)})
        if best is None or result.dev_score > best.dev_score:
            best = result
    best.trials = trials
    return best
