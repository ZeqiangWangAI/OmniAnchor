import numpy as np
from scipy.stats import spearmanr

from scripts.sensitivity_statistics import pair_distance_agreement, interval_or_undefined, template_clusters


def test_repeated_bootstrap_nodes_do_not_create_self_edge_agreement():
    a = np.array([[0., 1., 2.], [1., 0., 3.], [2., 3., 0.]])
    b = np.array([[0., 3., 2.], [3., 0., 1.], [2., 1., 0.]])
    # Node0 occurs twice: its edges are duplicated but the artificial0-to-0 pair is omitted.
    expected = spearmanr([1., 2., 1., 2., 3.], [3., 2., 3., 2., 1.]).statistic
    assert np.isclose(pair_distance_agreement(a, b, [0, 0, 1, 2]), expected)
    assert np.isclose(expected, -1.)


def test_undefined_sensitivity_is_not_zero_or_fabricated_interval():
    result = interval_or_undefined(lambda index: np.nan, ["a", "b"])
    assert result["estimate"] is None and result["lower"] is None
    result = interval_or_undefined(lambda index: .9, ["same", "same"])
    assert result["estimate"] == .9 and result["lower"] is None


def test_template_stability_is_invariant_to_template_translation():
    from threadpoolctl import threadpool_limits
    values = np.array([[-4., -3.], [-3., -4.], [3., 4.], [4., 3.]])
    cube = np.stack([values, values+5, values-2], axis=2)
    with threadpool_limits(limits=1):
        result, labels = template_clusters(cube, ["a", "b", "c", "d"], n_resamples=10)
    assert result["template_mean_ari"] == 1.
    assert result["template_ari_interval"] == [1., 1.]
    assert labels.shape == (4, 4)
