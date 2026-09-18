"""Bounded PMPO-inspired selection using training labels and repeated bridges.

The scorer/operator are callables so proposing wording never requires exposing
examples to a generation service. The default objective is a design choice,
not a psychometric validity theorem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, islice
from typing import Callable, Sequence
import re

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

from .provenance import stable_hash
from .reliability import cronbach_alpha
from .types import Bridge


@dataclass(frozen=True)
class OptimizationData:
    labels: np.ndarray
    sample_ids: tuple[str, ...]
    anchor_ids: tuple[str, ...]
    split: str = "train"
    group_ids: tuple[str, ...] = ()
    reference_group_ids: tuple[str, ...] = ()

    def validate(self):
        labels = np.asarray(self.labels)
        if self.split != "train":
            raise ValueError("Bridge optimization only accepts a declared training split.")
        if labels.shape != (len(self.sample_ids), len(self.anchor_ids)):
            raise ValueError("Optimization labels and IDs differ in shape.")
        for name, ids in [("sample", self.sample_ids), ("anchor", self.anchor_ids)]:
            if not ids or len(set(ids)) != len(ids):
                raise ValueError(f"Optimization {name} IDs must be nonempty and unique.")
        if not np.isin(labels, [0, 1]).all():
            raise ValueError("Optimization requires complete binary labels.")
        if self.group_ids and len(self.group_ids) != len(self.sample_ids):
            raise ValueError("One group ID is required per optimization sample.")
        if set(self.group_ids) & set(self.reference_group_ids):
            raise ValueError("Reference and optimization groups overlap.")


@dataclass(frozen=True)
class OptimizationConfig:
    subset_size: int = 3
    rounds: int = 2
    offspring_per_round: int = 4
    max_unique_bridges: int = 16
    min_positive: int = 5
    min_negative: int = 5
    validity_weight: float = 0.8
    seed: int = 42
    protected_relation_phrase: str | None = None
    forbidden_surfaces: tuple[str, ...] = ()
    objective: str = "reference_guided"


@dataclass
class BridgeArtifact:
    bridges: tuple[Bridge, ...]
    status: str
    trace: list[dict] = field(default_factory=list)
    manifest: dict = field(default_factory=dict)


def _lexical_diversity(bridges: Sequence[Bridge]) -> float:
    words = [set(re.findall(r"\w", b.prefix.lower())) if b.language.startswith("zh")
             else set(re.findall(r"\w+", b.prefix.lower())) for b in bridges]
    return float(np.mean([1 - len(a & b) / max(1, len(a | b))
                          for a, b in combinations(words, 2)]))


def _objective(bridges, cache, labels, eligible, weight):
    tensors = np.stack([cache[b.id] for b in bridges], axis=1)
    mean_scores = tensors.mean(axis=1)
    aps, correlations = [], []
    for j in np.flatnonzero(eligible):
        aps.append(average_precision_score(labels[:, j], mean_scores[:, j]))
        pair_scores = []
        for b, c in combinations(range(len(bridges)), 2):
            x, y = tensors[:, b, j], tensors[:, c, j]
            if np.ptp(x) == 0 or np.ptp(y) == 0:
                return None
            pair_scores.append(float(spearmanr(x, y).statistic))
        correlations.append(float(np.mean(pair_scores)))
    v = float(np.mean(aps))
    r = float((1 + np.median(correlations)) / 2)
    return {"objective": weight * v + (1 - weight) * r, "validity_macro_ap": v,
            "reliability": r, "lexical_diversity": _lexical_diversity(bridges)}


def _candidate_rejection(bridge, relation, language, config):
    if not isinstance(bridge, Bridge):
        return "invalid_candidate_type"
    if (bridge.relation, bridge.language) != (relation, language):
        return "relation_or_language_changed"
    if not bridge.prefix.endswith("\n") or len(bridge.prefix) > 512:
        return "invalid_delimiter_or_length"
    prefix = bridge.prefix.casefold()
    if any(token in prefix for token in ("<|", "[mask]", "<think>", "</think>",
                                         "[inst]", "[/inst]", "<<sys>>")) or any(
            ord(char) < 32 and char not in "\n\r\t" for char in prefix):
        return "control_token"
    if config.protected_relation_phrase and config.protected_relation_phrase not in bridge.prefix:
        return "protected_relation_changed"
    for surface in config.forbidden_surfaces:
        if not surface:
            continue
        text = surface.casefold()
        if re.search(r"[\u3400-\u9fff]", text):
            found = text in prefix
        else:
            left = r"(?<!\w)" if text[0].isalnum() or text[0] == "_" else ""
            right = r"(?!\w)" if text[-1].isalnum() or text[-1] == "_" else ""
            found = re.search(left + re.escape(text) + right, prefix) is not None
        if found:
            return "answer_surface_inserted"
    return None


def optimize_bridges(
    seed_bridges: Sequence[Bridge], train_data: OptimizationData,
    config: OptimizationConfig = OptimizationConfig(), *,
    score_bridge: Callable[[Bridge], np.ndarray],
    operator: Callable[[tuple[Bridge, ...], int, int], Sequence[Bridge]] | None = None,
) -> BridgeArtifact:
    """Score each unique bridge once, then enumerate bounded candidate subsets.

    score_bridge returns [training samples, anchors] in the exact data ID order,
    preferably reference-z calibrated on a separate fixed reference set. operator
    only receives selected bridges, requested count and deterministic round seed.
    """
    train_data.validate()
    for name in ["subset_size", "rounds", "offspring_per_round", "max_unique_bridges",
                 "min_positive", "min_negative", "seed"]:
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer.")
    if config.subset_size < 2 or config.rounds < 0 or config.offspring_per_round < 0:
        raise ValueError("Invalid search budget.")
    if not config.subset_size <= config.max_unique_bridges <= 16:
        raise ValueError("Search budget must fit the subset and cannot exceed 16 scored bridges.")
    if not 0 <= config.validity_weight <= 1:
        raise ValueError("validity_weight must lie in [0,1].")
    if config.objective not in {"reference_guided", "validity", "reliability", "alpha", "random"}:
        raise ValueError("Unknown bridge selection objective.")
    if config.min_positive < 1 or config.min_negative < 1:
        raise ValueError("Eligibility requires positive and negative examples.")
    if not seed_bridges:
        raise ValueError("At least one fixed seed bridge is required.")
    if not all(isinstance(b, Bridge) for b in seed_bridges):
        raise ValueError("Seed candidates must be Bridge objects.")
    if len(seed_bridges) > config.max_unique_bridges:
        raise ValueError("Seed pool exceeds the search budget.")
    if len({(b.relation, b.language) for b in seed_bridges}) != 1:
        raise ValueError("Optimization must preserve one relation and language.")
    if len({b.id for b in seed_bridges}) != len(seed_bridges) or len({b.prefix for b in seed_bridges}) != len(seed_bridges):
        raise ValueError("Seed bridges need unique IDs and prefixes.")
    relation, language = seed_bridges[0].relation, seed_bridges[0].language
    for bridge in seed_bridges:
        reason = _candidate_rejection(bridge, relation, language, config)
        if reason:
            raise ValueError(f"Invalid fixed seed bridge {bridge.id}: {reason}")
    labels = np.array(train_data.labels, dtype=int, copy=True)
    eligible = ((labels.sum(0) >= config.min_positive)
                & ((1-labels).sum(0) >= config.min_negative))
    fallback = tuple(seed_bridges[:config.subset_size])
    manifest = {"mode": config.objective, "config": vars(config), "split": "train",
                "sample_ids": train_data.sample_ids, "anchor_ids": train_data.anchor_ids,
                "eligible_anchor_ids": [a for a, use in zip(train_data.anchor_ids, eligible) if use],
                "data_hash": stable_hash({"ids": train_data.sample_ids, "labels": labels.tolist()}),
                "semantic_equivalence": "rules_do_not_prove_equivalence",
                "scoring_attempt_count": 0, "scored_bridge_count": 0,
                "answer_surface_guard": bool(config.forbidden_surfaces)}
    if not eligible.any():
        return BridgeArtifact(fallback, "no_eligible_anchors", manifest=manifest)
    cache, accepted, trace = {}, {}, []
    attempted_ids, attempted_prefixes = set(), set()
    selection_rng = np.random.default_rng(config.seed)

    def accept(bridge, generation):
        reason = _candidate_rejection(bridge, relation, language, config)
        if reason is None and (bridge.id in attempted_ids or bridge.prefix in attempted_prefixes):
            reason = "duplicate"
        if reason:
            trace.append({"generation": generation,
                          "bridge": bridge.model_dump() if isinstance(bridge, Bridge) else None,
                          "rejected": reason})
            return False
        if len(attempted_ids) >= config.max_unique_bridges:
            return False
        attempted_ids.add(bridge.id)
        attempted_prefixes.add(bridge.prefix)
        try:
            values = np.array(score_bridge(bridge), dtype=float, copy=True)
        except Exception as exc:
            trace.append({"generation": generation, "bridge": bridge.model_dump(),
                          "rejected": "score_failed", "error_type": type(exc).__name__})
            return False
        if values.shape != labels.shape or not np.isfinite(values).all():
            trace.append({"generation": generation, "bridge": bridge.model_dump(),
                          "rejected": "invalid_scores"})
            return False
        accepted[bridge.id] = bridge
        cache[bridge.id] = values
        return True

    def select(pool):
        choices = []
        for subset in combinations(sorted(pool, key=lambda b: b.id), config.subset_size):
            result = _objective(subset, cache, labels, eligible, config.validity_weight)
            if result:
                if config.objective == "validity":
                    result["objective"] = result["validity_macro_ap"]
                elif config.objective == "reliability":
                    result["objective"] = result["reliability"]
                elif config.objective == "alpha":
                    alphas = [cronbach_alpha(np.column_stack([cache[b.id][:, j] for b in subset]))["alpha"]
                              for j in np.flatnonzero(eligible)]
                    if not np.isfinite(alphas).all():
                        continue
                    result["objective"] = result["median_within_anchor_alpha"] = float(np.median(alphas))
                elif config.objective == "random":
                    result["objective"] = 0.0
                choices.append((subset, result))
        if not choices:
            return None
        if config.objective == "random":
            return choices[int(selection_rng.integers(len(choices)))]
        return min(choices, key=lambda p: (-p[1]["objective"], -p[1]["lexical_diversity"],
                                           tuple(b.id for b in p[0])))

    for bridge in seed_bridges:
        accept(bridge, 0)
    best = select(list(accepted.values()))
    if best is None:
        manifest.update(scoring_attempt_count=len(attempted_ids), scored_bridge_count=len(cache),
                        failed_bridge_count=len(attempted_ids) - len(cache))
        status = "undefined_reliability" if len(accepted) >= config.subset_size else "insufficient_valid_bridges"
        return BridgeArtifact(tuple(accepted.values())[:config.subset_size], status, trace, manifest)
    selected, metrics = best
    trace.append({"generation": 0, "selected": [b.id for b in selected], **metrics})
    for generation in range(1, config.rounds + 1):
        if operator is None or len(attempted_ids) >= config.max_unique_bridges:
            break
        requested = min(config.offspring_per_round, config.max_unique_bridges - len(attempted_ids))
        if not requested:
            break
        try:
            proposals = list(islice(operator(tuple(selected), requested, config.seed + generation), requested))
        except Exception as exc:
            trace.append({"generation": generation, "rejected": "operator_failed",
                          "error_type": type(exc).__name__})
            break
        children = []
        for bridge in proposals:
            if accept(bridge, generation):
                children.append(bridge)
        best = select(list(selected) + children)
        if best is not None:
            selected, metrics = best
        trace.append({"generation": generation, "selected": [b.id for b in selected], **metrics})
    manifest["scored_bridge_count"] = len(cache)
    manifest["scoring_attempt_count"] = len(attempted_ids)
    manifest["failed_bridge_count"] = len(attempted_ids) - len(cache)
    manifest["artifact_id"] = stable_hash({"bridges": [b.model_dump() for b in selected],
                                            "manifest": manifest})
    return BridgeArtifact(tuple(selected), "ok", trace, manifest)
