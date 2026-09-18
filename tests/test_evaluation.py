import numpy as np
import pytest

from vlanchor.evaluation import (
    compare_networks, evaluate_candidates, evaluate_multilabel, evaluate_vad,
    fit_linear_probe, group_bootstrap, retrieval_metrics,
)


def test_per_anchor_ap_and_within_sample_ap_have_different_axes():
    y = np.array([[1, 0], [0, 1]])
    scores = np.array([[1, 100], [0, 101]])
    assert evaluate_multilabel(y, scores)["macro_ap"] == 1
    within = evaluate_candidates({"x": y[0], "y": y[1]}, {"x": scores[0], "y": scores[1]})
    assert within["mean_instance_ap"] == 0.75


def test_fixed_reference_translation_cannot_change_macro_ap():
    y = [[1, 0], [0, 1], [1, 1]]
    scores = np.array([[2, 0], [0, 3], [3, 4]])
    raw, centered = evaluate_multilabel(y, scores), evaluate_multilabel(y, scores - [100, -100])
    assert raw["macro_ap"] == centered["macro_ap"]
    assert raw["per_anchor_ap"] == centered["per_anchor_ap"]
    assert raw["micro_ap"] != centered["micro_ap"]  # Micro pools scores across anchors.


def test_missing_scores_are_not_silently_replaced():
    with pytest.raises(ValueError, match="finite"):
        evaluate_multilabel([[1, 0]], [[1, np.nan]])
    result = evaluate_multilabel([[1, 0], [0, 0]], [[2, 2], [1, 1]], ["observed", "absent"])
    assert result["excluded_anchors"] == ["absent"]


def test_vad_reports_constant_dimensions_as_undefined():
    result = evaluate_vad([[1, 1], [2, 1], [3, 1]], [[3, 0], [2, 1], [1, 2]], ("V", "A"))
    assert result["spearman"]["V"] == -1
    assert result["undefined_dimensions"] == ["A"]


def test_retrieval_success_is_distinct_from_fraction_recall():
    result = retrieval_metrics([[1, 1, 0]], [[3, 2, 1]], ks=[1, 2])
    assert result["success@1"] == 1
    assert result["recall@1"] == 0.5
    assert result["recall@2"] == 1


def test_group_bootstrap_resamples_intact_groups_and_is_deterministic():
    observed = []
    values = np.array([1, 1, 2, 2, 4, 4])
    def statistic(indices):
        for i in (0, 2, 4):
            assert np.sum(indices == i) == np.sum(indices == i + 1)
        observed.append(len(indices))
        return values[indices].mean()
    result = group_bootstrap(statistic, ["a", "a", "b", "b", "c", "c"], n_resamples=50)
    other = group_bootstrap(statistic, ["a", "a", "b", "b", "c", "c"], n_resamples=50)
    assert result == other
    assert result["independent_groups"] == 3
    assert result["lower"] <= result["estimate"] <= result["upper"]


def test_network_excludes_diagonal_and_zero_edges():
    network = np.array([[100, 1, 0], [1, 20, 2], [0, 2, 30]], dtype=float)
    result = compare_networks(network, network, top_k=2)
    assert result["edge_weight_spearman"] == 1
    assert result["top_k_edge_jaccard"] == 1
    assert result["compared_edge_count"] == 3
    assert compare_networks(np.zeros((3, 3)), np.zeros((3, 3)))["top_k_edge_jaccard"] is None


def test_linear_probe_scaler_never_fits_development_rows():
    train = np.arange(20, dtype=float).reshape(-1, 1)
    labels = (train > 9).astype(int)
    dev = np.array([[-10], [100]])
    result = fit_linear_probe(train, labels, dev, [[0], [1]], grid=[0.1, 1])
    assert result.scaler.mean_[0] == 9.5
    assert len(result.trials) == 2
    assert result.predict_scores(dev).shape == (2, 1)
    assert result.dev_score == 1


def test_regression_probe_and_constant_multilabel_columns():
    train = np.arange(12, dtype=float).reshape(-1, 1)
    ridge = fit_linear_probe(train, train * 2, [[12], [13]], [[24], [26]], task="regression")
    assert ridge.best_parameter == 0.01
    assert [trial["parameter"] for trial in ridge.trials] == [0.01, 0.1, 1, 10, 100]
    y = np.column_stack([train[:, 0] > 5, np.ones(12)])
    probe = fit_linear_probe(train, y, [[2], [8]], [[0, 1], [1, 1]], grid=[1])
    assert np.all(probe.predict_scores([[0], [10]])[:, 1] == 1)


@pytest.mark.parametrize("task", ["multilabel", "multiclass"])
def test_logistic_probe_defaults_match_the_approved_dev_grid(task):
    train = np.arange(12, dtype=float).reshape(-1, 1)
    labels = (train > 5).astype(int)
    dev_labels = np.array([[0], [1]])
    if task == "multiclass":
        labels, dev_labels = labels[:, 0], dev_labels[:, 0]
    result = fit_linear_probe(train, labels, [[2], [8]], dev_labels, task=task)
    assert [trial["parameter"] for trial in result.trials] == [0.01, 0.1, 1, 10]
