"""Check every saved DWUG adjacency matrix without pruning or imputing edges."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.analysis/'manifest.json').read_text())
    expected_ids = set(manifest['sample_ids'])
    shifts = pd.read_csv(args.analysis/'per-target-shift.csv')
    rows, hashes = [], {}
    for method in shifts.method.unique():
        seen = set()
        for row in shifts[shifts.method == method].itertuples():
            path = args.analysis/method/(row.target+'-adjacency.npz')
            saved = np.load(path)
            ids, matrix = saved['sample_ids'].tolist(), saved['adjacency']
            n = len(ids)
            if len(set(ids)) != n or seen.intersection(ids) or n != row.n_rows_0+row.n_rows_1:
                raise ValueError('Missing, repeated or cross-target nodes.')
            if matrix.shape != (n, n) or not np.isfinite(matrix).all():
                raise ValueError('Invalid complete matrix.')
            error = float(np.max(np.abs(matrix-matrix.T)))
            diagonal_error = float(np.max(np.abs(np.diag(matrix)-1)))
            if error > 1e-12 or diagonal_error > 1e-12 or matrix.min() < -1 or matrix.max() > 1:
                raise ValueError('Invalid cosine adjacency.')
            seen.update(ids)
            rows.append(dict(method=method, target=row.target, nodes=n, undirected_edges=n*(n-1)//2,
                symmetry_error=error, diagonal_error=diagonal_error))
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        if seen != expected_ids:
            raise ValueError('Network population differs from frozen analysis.')
    pd.DataFrame(rows).to_csv(args.output/'network-integrity.csv', index=False)
    (args.output/'manifest.json').write_text(json.dumps({'status': 'passed', 'networks': len(rows),
        'nodes_per_method': len(expected_ids), 'source_sha256': hashes,
        'scope': 'Complete stored cosine matrices and node coverage; not human network validity or independent reconstruction from raw scores'}, indent=2))


if __name__ == '__main__':
    main()
