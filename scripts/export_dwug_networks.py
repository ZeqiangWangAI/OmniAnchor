"""Export complete saved DWUG networks and aligned, partially observed human graphs."""
import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from omnianchor.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['analysis', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    root = args.analysis
    assert json.loads((root/'events.jsonl').read_text().splitlines()[-1])['status'] == 'completed'
    gold_path = root/'aggregated-pair-gold.csv'
    gold = pd.read_csv(gold_path)
    shifts = pd.read_csv(root/'per-target-shift.csv')
    args.output.mkdir(parents=True, exist_ok=False)
    hashes, counts, human_orders = {str(gold_path): file_hash(gold_path)}, [], {}
    for row in shifts.itertuples():
        path = root/row.method/(row.target+'-adjacency.npz')
        with np.load(path, allow_pickle=False) as saved:
            ids, adjacency = saved['sample_ids'].tolist(), saved['adjacency']
        n = len(ids)
        assert len(set(ids)) == n and adjacency.shape == (n, n) and np.isfinite(adjacency).all()
        folder = args.output/row.method
        folder.mkdir(exist_ok=True)
        i, j = np.triu_indices(n, 1)
        edges = pd.DataFrame({'source': np.array(ids)[i], 'target': np.array(ids)[j], 'weight': adjacency[i, j]})
        edges.to_csv(folder/(row.target+'-edges.csv'), index=False)
        graph = nx.Graph()
        graph.add_nodes_from(ids)
        graph.add_weighted_edges_from(edges.itertuples(index=False, name=None))
        nx.write_graphml(graph, folder/(row.target+'.graphml'))
        if row.target not in human_orders:
            human_orders[row.target] = ids
            human = np.full((n, n), np.nan)
            positions = {name: k for k, name in enumerate(ids)}
            pairs = gold[gold.lemma == row.target]
            for pair in pairs.itertuples():
                a, b = positions[pair.left], positions[pair.right]
                assert a != b and np.isnan(human[a, b]) and np.isfinite(pair.judgment)
                human[a, b] = human[b, a] = pair.judgment
            human_folder = args.output/'human'
            human_folder.mkdir(exist_ok=True)
            np.savez_compressed(human_folder/(row.target+'-adjacency.npz'), sample_ids=np.array(ids), adjacency=human)
            pairs.to_csv(human_folder/(row.target+'-observed-edges.csv'), index=False)
        else:
            assert ids == human_orders[row.target]
        hashes[str(path)] = file_hash(path)
        counts.append({'method': row.method, 'target': row.target, 'nodes': n, 'edges': len(edges)})
    pd.DataFrame(counts).to_csv(args.output/'coverage.csv', index=False)
    (args.output/'manifest.json').write_text(json.dumps({'status': 'completed', 'networks': len(counts),
        'source_sha256': hashes, 'script_sha256': file_hash(Path(__file__)),
        'human_policy': 'Original ordinal relatedness, same node order; unobserved edges and diagonal NaN, never zero. No rescaling to cosine.',
        'scope': 'Format completion from frozen outputs; no new inference, tuning, bootstrap or human judgments.'}, indent=2))


if __name__ == '__main__':
    main()
