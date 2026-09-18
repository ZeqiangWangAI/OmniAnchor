import pytest

from scripts.verify_template_ari import ari


@pytest.mark.parametrize('left,right,expected', [
    ([0, 0, 1, 1], [7, 7, 3, 3], 1.0),
    ([0, 0, 1, 1], [0, 1, 0, 1], -0.5),
    ([0, 0, 0, 0], [5, 5, 5, 5], 1.0),
    ([0, 1, 2, 3], [9, 8, 7, 6], 1.0),
    ([0, 0, 0, 0], [0, 0, 1, 1], 0.0),
])
def test_pair_count_ari_known_partitions(left, right, expected):
    assert ari(left, right) == pytest.approx(expected, abs=1e-15)
