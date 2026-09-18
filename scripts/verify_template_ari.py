"""Recompute saved template ARI using pair counts, without sklearn metric calls."""
import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from vlanchor.provenance import file_hash


def ari(left, right):
    _, x = np.unique(left, return_inverse=True)
    _, y = np.unique(right, return_inverse=True)
    table = np.zeros((x.max()+1, y.max()+1), dtype=np.int64)
    np.add.at(table, (x, y), 1)
    pairs = lambda counts: np.sum(counts*(counts-1)//2)
    overlap, a, b = pairs(table), pairs(table.sum(axis=1)), pairs(table.sum(axis=0))
    total = len(x)*(len(x)-1)//2
    expected = a*b/total
    denominator = (a+b)/2-expected
    return 1.0 if denominator == 0 else float((overlap-expected)/denominator)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.analysis/'clusters.json'
    records = json.loads(source.read_text())
    checked, hashes = {}, {str(source): file_hash(source)}
    for name, record in records.items():
        assert record['status'] == 'ok'
        path = args.analysis/(name+'-clusters.npz')
        with np.load(path, allow_pickle=False) as data:
            assert data['rows'].tolist() == ['three_bridge_mean', 'bridge0', 'bridge1', 'bridge2']
            labels, ids = data['labels'], data['sample_ids']
        assert labels.shape == (4, len(ids)) and len(set(ids)) == len(ids)
        values = [ari(labels[i], labels[j]) for i, j in combinations(range(1, 4), 2)]
        actual = float(np.mean(values))
        error = abs(actual-record['template_mean_ari'])
        assert error < 1e-12
        checked[name] = {'nodes': len(ids), 'pairwise_ari': values, 'mean': actual, 'error': error}
        hashes[str(path)] = file_hash(path)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'verification.json').write_text(json.dumps({'status': 'passed', 'instruments': checked,
        'source_sha256': hashes, 'script_sha256': file_hash(Path(__file__)),
        'scope': 'Template ARI point estimates from saved cluster assignments; no cluster refit, bootstrap reconstruction or human-cluster validity claim.'}, indent=2))


if __name__ == '__main__':
    main()
