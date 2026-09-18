import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cdist

from vlanchor.analysis import (bh_fdr, cluster, compare_groups, concept_network, energy_distance,
                               independent_column_null, pca, sample_distance_graph, semantic_shift)
from vlanchor.errors import MissingScores
from vlanchor.types import MeasurementMatrix


def matrix(values):
    values = np.asarray(values, dtype=float)
    return MeasurementMatrix(values, tuple(f"s{i}" for i in range(len(values))),
                             tuple(f"a{j}" for j in range(values.shape[1])))


def test_translation_distances_clustering_pca_and_column_correlation():
    rng = np.random.default_rng(10)
    a = matrix(rng.normal(size=(20, 3)))
    b = matrix(a.values - [2, 40, -4])
    np.testing.assert_allclose(cdist(a.values, a.values), cdist(b.values, b.values), atol=1e-13)
    assert cluster(a, k=3)["labels"] == cluster(b, k=3)["labels"]
    np.testing.assert_allclose(concept_network(a)["correlation"], concept_network(b)["correlation"])
    np.testing.assert_allclose(pca(a)["coordinates"].iloc[:, 1:], pca(b)["coordinates"].iloc[:, 1:])


def test_energy_exact_blocked_and_equal_mean_counterexample():
    a, b = np.array([[-1.], [1.]]), np.zeros((3, 1))
    assert np.linalg.norm(a.mean(0) - b.mean(0)) == 0
    assert energy_distance(a, b, block_size=1) == pytest.approx(1)
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(13, 4)), rng.normal(size=(7, 4))
    expected = 2 * cdist(a, b).mean() - cdist(a, a).mean() - cdist(b, b).mean()
    assert energy_distance(a, b, block_size=3) == pytest.approx(expected)
    assert energy_distance(a, a, block_size=3) == pytest.approx(0, abs=1e-12)
    assert energy_distance(a + 3, b + 3) == pytest.approx(expected)


def test_duplicate_anchor_is_extra_geometric_weight():
    values = np.array([[0., 0.], [1., 0.], [0., 1.]])
    duplicated = np.column_stack([values, values[:, 0]])
    original = cdist(values, values) ** 2
    actual = cdist(duplicated, duplicated) ** 2
    np.testing.assert_allclose(actual, original + (values[:, 0, None] - values[None, :, 0]) ** 2)
    assert original[0, 1] == original[0, 2]
    assert actual[0, 1] > actual[0, 2]


def test_group_bootstrap_reproducible_and_no_implicit_group_choice():
    x = matrix([[-2, -1], [-1, -3], [1, 2], [3, 4]])
    first = compare_groups(x, ["a", "a", "b", "b"], n_bootstrap=30)
    second = compare_groups(x, ["a", "a", "b", "b"], n_bootstrap=30)
    pd.testing.assert_frame_equal(first["differences"], second["differences"])
    assert first["centroid_distance"] > 0
    with pytest.raises(ValueError):
        compare_groups(x, ["a", "b", "c", "c"])
    assert np.isnan(compare_groups(x, ["a", "b", "b", "b"], n_bootstrap=10)["distance_ci_lower"])


def test_default_source_groups_are_not_counted_as_independent_rows():
    x = matrix([[0], [1], [2], [3], [10], [11], [12], [13]])
    x.manifest["samples"] = [{"id": f"s{i}", "group_id": "source_a" if i < 4 else "source_b"}
                             for i in range(8)]
    result = compare_groups(x, ["old"] * 4 + ["new"] * 4, n_bootstrap=30)
    assert result["n_a"] == result["n_b"] == 4
    assert result["n_units_a"] == result["n_units_b"] == 1
    assert result["bootstrap_replicates"] == 0
    assert result["uncertainty_status"] == "undefined_insufficient_sampling_units"
    assert result["differences"].ci_lower.isna().all()
    assert np.isnan(result["distance_ci_upper"])


def test_paired_source_bootstrap_preserves_source_membership_across_cohorts():
    # Every source moves by exactly one. Independent cohort draws would add
    # artificial uncertainty by comparing source 1 against source 2.
    x = matrix([[0], [0], [10], [10], [1], [1], [11], [11]])
    units = ["source_1", "source_1", "source_2", "source_2"] * 2
    result = compare_groups(x, ["old"] * 4 + ["new"] * 4,
                            sampling_units=units, n_bootstrap=100)
    assert result["bootstrap_design"] == "paired_clusters"
    assert result["n_units_a"] == result["n_units_b"] == result["n_shared_units"] == 2
    assert result["distance_ci_lower"] == result["distance_ci_upper"] == 1
    assert result["differences"].ci_lower.iloc[0] == result["differences"].ci_upper.iloc[0] == 1
    mapping = dict(zip(x.sample_ids, units))
    mapped = compare_groups(x, ["old"] * 4 + ["new"] * 4,
                            sampling_units=mapping, n_bootstrap=100)
    pd.testing.assert_frame_equal(result["differences"], mapped["differences"])


def test_partial_source_overlap_requires_an_explicit_study_design():
    x = matrix([[0], [1], [2], [3]])
    units = ["source_1", "shared", "shared", "source_2"]
    with pytest.raises(ValueError, match="partially overlap"):
        compare_groups(x, ["old", "old", "new", "new"], sampling_units=units)
    with pytest.raises(ValueError, match="partially overlap"):
        semantic_shift(x, target_ids=["bank"] * 4, periods=["old", "old", "new", "new"],
                       sampling_units=units)


def test_shift_target_periods_bootstrap_permutation_and_metadata():
    x = matrix([[-1], [1], [0], [0]])
    result = semantic_shift(x, target_ids=["word"] * 4, periods=["old"] * 2 + ["new"] * 2,
                            n_bootstrap=20, n_permutations=19)["results"].iloc[0]
    assert result.centroid_distance == 0
    assert result.energy_distance == 1
    assert 1 / 20 <= result.permutation_pvalue <= 1
    x.manifest["samples"] = [{"id": f"s{i}", "target": {"lemma": "word"},
                               "time": "old" if i < 2 else "new"} for i in range(4)]
    implicit = semantic_shift(x, n_bootstrap=0)["results"].iloc[0]
    assert implicit.energy_distance == 1
    missing = semantic_shift(x, target_ids=["word"] * 4, periods=["old"] * 4,
                             n_bootstrap=0)["results"].iloc[0]
    assert missing.status == "requires_two_periods"


def test_shift_uses_paired_source_groups_and_group_permutations():
    x = matrix([[0], [0], [10], [10], [1], [1], [11], [11]])
    units = ["source_1", "source_1", "source_2", "source_2"] * 2
    x.manifest["samples"] = [{"id": sample_id, "group_id": unit,
                               "target": {"lemma": "bank"}, "time": "old" if i < 4 else "new"}
                              for i, (sample_id, unit) in enumerate(zip(x.sample_ids, units))]
    result = semantic_shift(x, n_bootstrap=50, n_permutations=19)
    row = result["results"].iloc[0]
    assert row.n_0 == row.n_1 == 4
    assert row.n_units_0 == row.n_units_1 == row.n_shared_units == 2
    assert row.bootstrap_design == "paired_clusters"
    assert row.centroid_ci_lower == row.centroid_ci_upper == 1
    assert row.permutations_run == 19
    assert 1 / 20 <= row.permutation_pvalue <= 1
    repeated = semantic_shift(x, n_bootstrap=50, n_permutations=19)
    pd.testing.assert_frame_equal(result["results"], repeated["results"])


def test_shift_one_source_is_not_resampled_as_many_contexts():
    x = matrix([[-1], [1], [0], [0]])
    x.manifest["samples"] = [{"id": f"s{i}", "group_id": "one_source"} for i in range(4)]
    result = semantic_shift(x, target_ids=["bank"] * 4, periods=["old"] * 2 + ["new"] * 2,
                            n_bootstrap=50, n_permutations=19)["results"].iloc[0]
    assert result.energy_distance == 1
    assert result.n_units_0 == result.n_units_1 == 1
    assert result.bootstrap_replicates == result.permutations_run == 0
    assert np.isnan(result.centroid_ci_lower) and np.isnan(result.permutation_pvalue)


def test_explicit_sampling_unit_coverage_is_required():
    x = matrix([[0], [1], [2], [3]])
    with pytest.raises(ValueError, match="sampling_unit"):
        compare_groups(x, ["a", "a", "b", "b"], sampling_units={"s0": "one"})


def test_network_constant_columns_and_independent_column_null():
    rng = np.random.default_rng(0)
    shared = rng.normal(size=1000)
    x = matrix(np.column_stack([shared, shared, np.ones(1000)]))
    graph = concept_network(x)
    assert graph["correlation"].iloc[0, 1] == pytest.approx(1)
    assert graph["correlation"].iloc[2].isna().all()
    row_permuted = matrix(x.values[rng.permutation(len(shared))])
    np.testing.assert_allclose(graph["correlation"], concept_network(row_permuted)["correlation"], equal_nan=True)
    null = independent_column_null(x)
    for j in range(3):
        np.testing.assert_array_equal(np.sort(null.values[:, j]), np.sort(x.values[:, j]))
    assert abs(concept_network(null)["correlation"].iloc[0, 1]) < 0.1


def test_distance_graph_bh_and_degenerate_inputs():
    x = matrix([[0, 0], [1, 0], [0, 2]])
    graph = sample_distance_graph(x, k=1)
    assert len(graph["edges"]) == 2
    assert (graph["edges"].distance > 0).all()
    np.testing.assert_allclose(bh_fdr([0.01, 0.04, 0.03, np.nan]), [0.03, 0.04, 0.04, np.nan])
    assert pca(matrix([[1, 1], [1, 1]]))["status"] == "undefined_variance"
    assert sample_distance_graph(matrix([[1]]))["edges"].empty
    with pytest.raises(ValueError):
        cluster(matrix([[1], [1]]), k=2)
    with pytest.raises(MissingScores):
        concept_network(matrix([[np.nan], [1]]))
    with pytest.raises(ValueError):
        bh_fdr([1.1])
