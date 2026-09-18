import numpy as np
import pytest

from scripts.build_vatex_retrieval import cosine_scores


def test_cosine_joins_rows_and_preserves_shared_coordinate_permutation():
    values = np.array([[0., 2.], [3., 0.], [1., 1.], [4., 0.]])
    ids = ["v2", "c1", "c2", "v1"]
    expected = np.array([[1., 0.], [1/np.sqrt(2), 1/np.sqrt(2)]])
    np.testing.assert_allclose(cosine_scores(values, ids, ["c1", "c2"], ["v1", "v2"]), expected)
    np.testing.assert_allclose(cosine_scores(values[:, ::-1], ids, ["c1", "c2"], ["v1", "v2"]), expected)


def test_cosine_does_not_drop_missing_or_undefined_queries():
    with pytest.raises(ValueError, match="Missing"):
        cosine_scores([[1., 2.]], ["v"], ["c"], ["v"])
    with pytest.raises(ValueError, match="Zero feature"):
        cosine_scores([[1., 2.], [0., 0.]], ["v", "c"], ["c"], ["v"])
    with pytest.raises(ValueError, match="unique row"):
        cosine_scores([[1., 2.], [3., 4.]], ["v", "v"], ["c"], ["v"])
