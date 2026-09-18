import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('edge_export', Path(__file__).parents[1]/'scripts/export_dwug_edge_bootstrap.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_counts_match_explicit_document_resampling():
    units = ['a', 'a', 'b', 'c', 'd']
    periods = ['old', 'old', 'old', 'new', 'new']
    actual = module.endpoint_counts(units, periods)
    expected = np.zeros((5, 5), dtype=int)
    rng = np.random.default_rng(42)
    for _ in range(1000):
        chosen = set()
        for pool in [['c', 'd'], ['a', 'b']]:
            chosen.update(pool[i] for i in rng.integers(2, size=2))
        for i, a in enumerate(units):
            for j, b in enumerate(units):
                expected[i, j] += a in chosen and b in chosen
    np.testing.assert_array_equal(actual, expected)
    assert actual[0, 1] == actual[0, 0]
    assert actual[0, 2] < actual[0, 0]


def test_cross_period_sources_rejected():
    with pytest.raises(ValueError, match='crosses'):
        module.endpoint_counts(['a', 'a', 'b', 'c'], [0, 1, 0, 1])
