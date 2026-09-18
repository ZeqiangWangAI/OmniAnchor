"""Export source-bootstrap endpoint coverage for fixed complete DWUG sample graphs."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vlanchor.io import load_samples
from vlanchor.provenance import file_hash


def endpoint_counts(units, periods, draws=1000):
    units, periods = np.asarray(units), np.asarray(periods)
    blocks = {u: np.flatnonzero(units == u) for u in np.unique(units)}
    if any(len(set(periods[b])) != 1 for b in blocks.values()):
        raise ValueError('A source crosses periods.')
    pools = [[u for u, b in blocks.items() if periods[b[0]] == t] for t in sorted(set(periods))]
    if len(pools) != 2 or min(map(len, pools)) < 2:
        raise ValueError('Require two periods with at least two sources each.')
    present = np.zeros((draws, len(units)), dtype=np.int64)
    rng = np.random.default_rng(42)
    for draw in range(draws):
        for pool in pools:
            selected = rng.integers(len(pool), size=len(pool))
            for index in np.unique(selected):
                present[draw, blocks[pool[index]]] = 1
    return present.T @ present


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['analysis', 'samples', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.analysis/'manifest.json').read_text())
    if file_hash(args.samples) not in manifest['inputs'].values():
        raise ValueError('Sample input differs from frozen analysis.')
    samples = {s.id: s for s in load_samples(args.samples)}
    args.output.mkdir(parents=True, exist_ok=False)
    rows = pd.read_csv(args.analysis/'per-target-shift.csv')
    counts, orders, coverage, hashes = {}, {}, [], {}
    for row in rows.itertuples():
        source = args.analysis/row.method/(row.target+'-adjacency.npz')
        with np.load(source, allow_pickle=False) as saved:
            ids = saved['sample_ids'].tolist()
        if row.target not in counts:
            selected = [samples[i] for i in ids]
            counts[row.target] = endpoint_counts([s.metadata['sampling_unit'] for s in selected], [s.time for s in selected])
            orders[row.target] = ids
        if ids != orders[row.target]:
            raise ValueError('Methods have different node order.')
        i, j = np.triu_indices(len(ids), 1)
        both = counts[row.target][i, j]
        folder = args.output/row.method
        folder.mkdir(exist_ok=True)
        pd.DataFrame({'source': np.array(ids)[i], 'target': np.array(ids)[j],
            'endpoint_copresence_count': both, 'bootstrap_draws': 1000,
            'unconditional_full_edge_frequency': both/1000,
            'conditional_full_edge_frequency': np.where(both > 0, 1., np.nan)}).to_csv(folder/(row.target+'-edge-frequency.csv'), index=False)
        coverage.append({'method': row.method, 'target': row.target, 'edges': len(i)})
        hashes[str(source)] = file_hash(source)
    pd.DataFrame(coverage).to_csv(args.output/'coverage.csv', index=False)
    (args.output/'manifest.json').write_text(json.dumps({'seed': 42, 'draws': 1000,
        'samples_sha256': file_hash(args.samples), 'source_sha256': hashes,
        'script_sha256': file_hash(Path(__file__)), 'networks': len(coverage),
        'scope': 'Post-result completion export; fixed graph endpoint coverage, not semantic reliability or a new significance test. Source resampling within periods matches existing source_shift draw order. Conditional frequency undefined if endpoints never co-occur. No labels, tuning, sparsification or new model calls.'}, indent=2))


if __name__ == '__main__':
    main()
