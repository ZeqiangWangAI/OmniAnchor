from dataclasses import replace

import numpy as np
import pytest

from omnianchor.optimization import OptimizationConfig, OptimizationData, optimize_bridges
from omnianchor.types import Bridge


def training_data():
    labels = np.column_stack([np.arange(12) < 6, np.arange(12) % 2 == 0]).astype(int)
    return OptimizationData(labels, tuple(f"s{i}" for i in range(12)), ("a", "b"))


def seeds(n=3):
    return tuple(Bridge(id=f"seed{i}", prefix=f"Content associated with concept variant {i}:\n")
                 for i in range(n))


def correct_scores(data):
    return data.labels.astype(float) + np.arange(len(data.sample_ids))[:, None] * 0.001


def test_random_selection_ignores_label_values_and_alpha_uses_within_anchor_templates():
    data = training_data()
    pool = seeds(4)
    x = np.column_stack([np.arange(12), np.arange(12)**2]).astype(float)
    def scorer(bridge):
        return x * (100 if bridge.id == "seed3" else 1)
    alpha = optimize_bridges(pool, data, OptimizationConfig(rounds=0, objective="alpha"), score_bridge=scorer)
    assert [b.id for b in alpha.bridges] == ["seed0", "seed1", "seed2"]
    assert alpha.trace[-1]["median_within_anchor_alpha"] == pytest.approx(1)
    config = OptimizationConfig(rounds=0, objective="random")
    first = optimize_bridges(pool, data, config, score_bridge=scorer)
    second = optimize_bridges(pool, replace(data, labels=1-data.labels), config, score_bridge=scorer)
    assert first.bridges == second.bridges
    assert first.trace[-1]["objective"] == 0


@pytest.mark.parametrize("mode,key", [("validity", "validity_macro_ap"), ("reliability", "reliability")])
def test_ablation_objective_uses_only_the_declared_selection_metric(mode, key):
    data = training_data()
    result = optimize_bridges(seeds(4), data, OptimizationConfig(rounds=0, objective=mode),
                              score_bridge=lambda _: correct_scores(data))
    assert result.trace[-1]["objective"] == result.trace[-1][key]


def test_default_two_round_budget_scores_sixteen_unique_bridges_only():
    data, calls, requests = training_data(), [], []

    def scorer(bridge):
        calls.append(bridge.prefix)
        return correct_scores(data)

    def operator(selected, count, seed):
        requests.append((len(selected), count, seed))
        # Extra returned candidates must never cause extra score calls.
        return [Bridge(id=f"child{seed}_{i}", prefix=f"Content associated with proposal {seed} {i}:\n")
                for i in range(count + 5)]

    artifact = optimize_bridges(seeds(8), data, score_bridge=scorer, operator=operator)
    assert len(calls) == len(set(calls)) == 16
    assert artifact.manifest["scoring_attempt_count"] == artifact.manifest["scored_bridge_count"] == 16
    assert requests == [(3, 4, 43), (3, 4, 44)]
    assert len(artifact.bridges) == 3 and artifact.status == "ok"


def test_failed_duplicate_candidates_are_not_rescored_or_allowed_to_bypass_budget():
    data, calls = training_data(), []

    def scorer(bridge):
        calls.append((bridge.id, bridge.prefix))
        return correct_scores(data) if bridge.id.startswith("seed") else np.full(data.labels.shape, np.nan)

    def operator(selected, count, seed):
        return [Bridge(id="repeated_failure", prefix="Content associated with recurring proposal:\n")] + [
            Bridge(id=f"new{seed}_{i}", prefix=f"Content associated with new wording {seed} {i}:\n")
            for i in range(count - 1)]

    artifact = optimize_bridges(seeds(), data, OptimizationConfig(rounds=10),
                                score_bridge=scorer, operator=operator)
    assert len(calls) == len({prefix for _, prefix in calls}) <= 16
    assert sum(identifier == "repeated_failure" for identifier, _ in calls) == 1
    assert artifact.manifest["scored_bridge_count"] == 3
    assert artifact.manifest["failed_bridge_count"] == len(calls) - 3
    assert {b.id for b in artifact.bridges} == {b.id for b in seeds()}
    assert any(row.get("rejected") == "duplicate" for row in artifact.trace)


@pytest.mark.parametrize("candidate,reason", [
    (Bridge(id="child", prefix="Different relation:\n", relation="supports"), "relation_or_language_changed"),
    (Bridge(id="child", prefix="关联概念：\n", language="zh"), "relation_or_language_changed"),
    (Bridge(id="child", prefix="Content associated with [MASK]:\n"), "control_token"),
    (Bridge(id="child", prefix="Content associated with <think>:\n"), "control_token"),
    (Bridge(id="child", prefix="Content associated with <|assistant|>:\n"), "control_token"),
    (Bridge(id="child", prefix="Content associated with \x00:\n"), "control_token"),
    (Bridge(id="child", prefix="Content associated with FAIRNESS:\n"), "answer_surface_inserted"),
    (Bridge(id="child", prefix="Content supports a concept:\n"), "protected_relation_changed"),
    (Bridge(id="child", prefix="Content associated with a concept:"), "invalid_delimiter_or_length"),
    (Bridge(id="child", prefix="x" * 512 + "\n"), "invalid_delimiter_or_length"),
])
def test_invalid_offspring_never_reaches_score_callback(candidate, reason):
    data, calls = training_data(), []

    def scorer(bridge):
        calls.append(bridge.id)
        return correct_scores(data)

    config = OptimizationConfig(rounds=1, offspring_per_round=1,
                                protected_relation_phrase="associated with", forbidden_surfaces=("fairness",))
    artifact = optimize_bridges(seeds(), data, config, score_bridge=scorer,
                                operator=lambda *_: [candidate])
    assert calls == [bridge.id for bridge in seeds()]
    assert artifact.trace[-2]["rejected"] == reason
    assert {b.id for b in artifact.bridges} == set(calls)


def test_anchor_guard_matches_words_without_rejecting_english_substrings():
    data, calls = training_data(), []

    def scorer(bridge):
        calls.append(bridge.id)
        return correct_scores(data)

    allowed = Bridge(id="child", prefix="Content associated with a particular concept:\n")
    artifact = optimize_bridges(seeds(), data,
                                OptimizationConfig(rounds=1, forbidden_surfaces=("art",)),
                                score_bridge=scorer, operator=lambda *_: [allowed])
    assert "child" in calls and artifact.status == "ok"


def test_new_scorer_exception_and_operator_exception_keep_valid_seed_fallback():
    data = training_data()

    def scorer(bridge):
        if bridge.id == "child":
            raise RuntimeError("candidate scoring failed")
        return correct_scores(data)

    artifact = optimize_bridges(seeds(), data, score_bridge=scorer,
                                operator=lambda *_: [Bridge(id="child", prefix="New association:\n")])
    assert artifact.status == "ok"
    assert artifact.manifest["scoring_attempt_count"] == 4
    assert artifact.manifest["failed_bridge_count"] == 1
    assert {b.id for b in artifact.bridges} == {b.id for b in seeds()}
    assert any(row.get("rejected") == "score_failed" for row in artifact.trace)

    def failed_operator(*args):
        raise RuntimeError("proposal failed")

    result = optimize_bridges(seeds(), data, score_bridge=scorer, operator=failed_operator)
    assert result.status == "ok"
    assert result.trace[-1]["rejected"] == "operator_failed"


@pytest.mark.parametrize("invalid", [np.full((12, 2), np.nan), np.zeros((1, 1))])
def test_invalid_child_score_arrays_keep_seed_fallback(invalid):
    data = training_data()
    artifact = optimize_bridges(seeds(), data, OptimizationConfig(rounds=1),
                                score_bridge=lambda b: invalid if b.id == "child" else correct_scores(data),
                                operator=lambda *_: [Bridge(id="child", prefix="New association:\n")])
    assert artifact.status == "ok"
    assert artifact.manifest["failed_bridge_count"] == 1
    assert {b.id for b in artifact.bridges} == {b.id for b in seeds()}


def test_successful_prefix_or_identifier_aliases_are_not_rescored():
    data, calls = training_data(), []

    def scorer(bridge):
        calls.append(bridge.id)
        return correct_scores(data)

    aliases = [Bridge(id="alias", prefix=seeds()[0].prefix),
               Bridge(id="seed1", prefix="A changed prefix under an existing ID:\n")]
    artifact = optimize_bridges(seeds(), data, OptimizationConfig(rounds=1),
                                score_bridge=scorer, operator=lambda *_: aliases)
    assert len(calls) == 3
    assert sum(row.get("rejected") == "duplicate" for row in artifact.trace) == 2


@pytest.mark.parametrize("split", ["test", "dev", "validation", "external"])
def test_nontraining_split_is_rejected_before_any_callback(split):
    data = replace(training_data(), split=split)
    with pytest.raises(ValueError, match="training split"):
        optimize_bridges(seeds(), data, score_bridge=lambda _: pytest.fail("must not score"))


def test_reference_overlap_and_duplicate_anchor_coordinates_are_rejected():
    data = training_data()
    overlapping = replace(data, group_ids=("source",) * len(data.sample_ids), reference_group_ids=("source",))
    with pytest.raises(ValueError, match="overlap"):
        optimize_bridges(seeds(), overlapping, score_bridge=lambda _: pytest.fail("must not score"))
    with pytest.raises(ValueError, match="anchor IDs"):
        optimize_bridges(seeds(), replace(data, anchor_ids=("a", "a")),
                         score_bridge=lambda _: pytest.fail("must not score"))


def test_eligibility_requires_positive_and_negative_cases_per_anchor():
    data = training_data()
    labels = np.column_stack([data.labels[:, 0], np.ones(12), np.arange(12) == 0])
    data = replace(data, labels=labels, anchor_ids=("eligible", "no_negatives", "one_positive"))
    artifact = optimize_bridges(seeds(), data, OptimizationConfig(rounds=0),
                                score_bridge=lambda _: correct_scores(data))
    assert artifact.manifest["eligible_anchor_ids"] == ["eligible"]
    empty = replace(data, labels=np.ones_like(labels))
    result = optimize_bridges(seeds(), empty, score_bridge=lambda _: pytest.fail("must not score"))
    assert result.status == "no_eligible_anchors"
    assert result.manifest["scoring_attempt_count"] == 0


def test_consistently_wrong_predictions_have_high_reliability_but_low_validity():
    data = training_data()
    wrong = 1 - correct_scores(data)
    artifact = optimize_bridges(seeds(), data, OptimizationConfig(rounds=0), score_bridge=lambda _: wrong)
    metrics = artifact.trace[-1]
    assert metrics["reliability"] == pytest.approx(1)
    assert metrics["validity_macro_ap"] < 0.5
    assert metrics["objective"] < 0.6


def test_reused_callback_buffer_cannot_overwrite_earlier_cached_scores():
    data = training_data()
    buffer = np.empty(data.labels.shape)

    def scorer(bridge):
        buffer[:] = 1 - correct_scores(data) if bridge.id == "seed1" else correct_scores(data)
        return buffer

    artifact = optimize_bridges(seeds(), data, OptimizationConfig(rounds=0), score_bridge=scorer)
    assert artifact.trace[-1]["reliability"] == pytest.approx(1 / 3)
    assert artifact.trace[-1]["validity_macro_ap"] == pytest.approx(1)


def test_undefined_reliability_is_not_reported_as_perfect():
    data = training_data()
    artifact = optimize_bridges(seeds(), data, score_bridge=lambda _: np.ones(data.labels.shape))
    assert artifact.status == "undefined_reliability"
    assert not any("objective" in row or "reliability" in row for row in artifact.trace)
    assert artifact.manifest["scored_bridge_count"] == 3


@pytest.mark.parametrize("change", [
    {"max_unique_bridges": 17}, {"max_unique_bridges": 0}, {"max_unique_bridges": 2},
    {"max_unique_bridges": 16.5}, {"subset_size": 1}, {"subset_size": 17},
    {"rounds": -1}, {"rounds": 1.5}, {"offspring_per_round": -1},
    {"offspring_per_round": True}, {"min_positive": 0}, {"min_negative": 0},
    {"validity_weight": -0.1}, {"validity_weight": 1.1}, {"validity_weight": np.nan},
])
def test_invalid_config_bounds_fail_before_scoring(change):
    config = replace(OptimizationConfig(), **change)
    with pytest.raises(ValueError):
        optimize_bridges(seeds(), training_data(), config,
                         score_bridge=lambda _: pytest.fail("must not score"))


def test_seed_budget_and_invalid_seed_fallback_are_rejected():
    with pytest.raises(ValueError, match="exceeds"):
        optimize_bridges(seeds(4), training_data(), OptimizationConfig(max_unique_bridges=3),
                         score_bridge=lambda _: pytest.fail("must not score"))
    bad = (Bridge(id="bad", prefix="Unsafe <think>:\n"),) + seeds(2)
    with pytest.raises(ValueError, match="Invalid fixed seed"):
        optimize_bridges(bad, training_data(), score_bridge=lambda _: pytest.fail("must not score"))
    with pytest.raises(ValueError, match="unique IDs"):
        optimize_bridges((seeds()[0], seeds()[0], seeds()[1]), training_data(),
                         score_bridge=lambda _: pytest.fail("must not score"))
