import numpy as np
import pytest
from scipy.spatial.distance import cdist

from scripts.dwug_statistics import distances_from_indices, source_shift
from vlanchor.analysis import energy_distance


def test_reused_distances_match_exact_energy_with_repeated_source_rows():
    x = np.random.default_rng(42).normal(size=(9, 5))
    a, b = np.array([0, 1, 0, 1, 2]), np.array([4, 5, 6, 6, 7, 8])
    observed = distances_from_indices(x, cdist(x, x), a, b)
    np.testing.assert_allclose(observed, [np.linalg.norm(x[a].mean(0)-x[b].mean(0)), energy_distance(x[a], x[b])], atol=1e-12)


def test_source_shift_rejects_cross_period_documents_and_preserves_undefined_ci():
    x = np.arange(8.).reshape(4, 2)
    with pytest.raises(ValueError, match="both periods"):
        source_shift(x, ["1", "1", "2", "2"], ["fic_same", "fic_a", "fic_same", "fic_b"], n_bootstrap=3)
    result = source_shift(x, ["1", "1", "2", "2"], ["fic_a", "fic_a", "fic_b", "fic_b"], n_bootstrap=3)
    assert result["bootstrap_replicates"] == result["permutations_run"] == 0
    assert np.isnan(result["energy_ci_lower"])
