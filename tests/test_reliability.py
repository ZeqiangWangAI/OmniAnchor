import numpy as np
import pytest
from scipy.spatial.distance import cdist

from vlanchor.calibration import to_matrix
from vlanchor.reliability import bridge_reliability, cronbach_alpha, template_radii
from test_calibration import make_scores


def test_alpha_perfect_independent_and_undefined():
    repeated = np.tile(np.arange(4)[:, None], (1, 3))
    assert cronbach_alpha(repeated)["alpha"] == pytest.approx(1)
    independent = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    assert cronbach_alpha(independent)["alpha"] == pytest.approx(0)
    assert cronbach_alpha(np.ones((4, 3)))["status"] == "undefined_total_variance"
    assert cronbach_alpha(np.ones((4, 1)))["status"] == "undefined_insufficient"
    assert cronbach_alpha([[1, 2], [np.nan, 3]])["status"] == "undefined_nonfinite"
    with pytest.raises(ValueError):
        cronbach_alpha(to_matrix(make_scores([[-1, -2], [-3, -4]])))


def test_same_anchor_template_rank_consistency_and_constants():
    cube = np.array([[[-1, -5], [-2, -2]], [[-2, -6], [-2, -2]], [[-3, -7], [-2, -2]]])
    result = bridge_reliability(make_scores(cube))
    assert result["pairs"].iloc[0].spearman == pytest.approx(1)
    assert np.isnan(result["pairs"].iloc[1].spearman)
    assert result["anchors"].iloc[1].alpha_status == "undefined_total_variance"


def test_observed_template_sample_and_correlation_bounds():
    rng = np.random.default_rng(42)
    cube = rng.normal(size=(12, 3, 4)) - 10
    scores = make_scores(cube)
    result = template_radii(scores)
    radii = result["samples"].radius.to_numpy()
    mean = cube.mean(axis=2)
    average_distances = cdist(mean, mean)
    for b in range(cube.shape[2]):
        difference = np.abs(cdist(cube[:, :, b], cube[:, :, b]) - average_distances)
        assert np.all(difference <= radii[:, None] + radii[None, :] + 1e-12)
        correlation = np.corrcoef(cube[:, :, b], rowvar=False)
        for row in result["correlation_bounds"].itertuples():
            j, k = int(row.source[1:]), int(row.target[1:])
            assert row.lower - 1e-12 <= correlation[j, k] <= row.upper + 1e-12


def test_single_bridge_bounds_and_degenerate_column_explicit():
    scores = make_scores([[-1, -2], [-1, -4], [-1, -6]])
    result = template_radii(scores)
    np.testing.assert_allclose(result["samples"].radius, 0)
    assert np.isnan(result["anchors"].iloc[0].normalized_radius)
    assert result["anchors"].iloc[1].normalized_radius == pytest.approx(0)
    assert bridge_reliability(scores)["pairs"].empty
