"""Audit complete concept-network exports against their saved adjacency matrices."""
import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from omnianchor.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.analysis
    manifest = json.loads((root/'manifest.json').read_text())
    summaries = json.loads((root/'summary.json').read_text())
    if json.loads((root/'events.jsonl').read_text().splitlines()[-1])['status'] != 'completed':
        raise ValueError('Require a completed analysis.')
    nodes = manifest['nodes']
    gold = pd.read_csv(root/'human-full-adjacency.csv', index_col=0)
    assert gold.index.tolist() == gold.columns.tolist() == nodes
    upper = np.triu_indices(len(nodes), 1)
    expected = {(nodes[i], nodes[j]) for i, j in zip(*upper)}
    checks, sources = {}, [root/'manifest.json', root/'summary.json', root/'human-full-adjacency.csv']
    for method, summary in summaries.items():
        folder = root/method
        matrix = pd.read_csv(folder/'full-adjacency.csv', index_col=0)
        edges = pd.read_csv(folder/'edges.csv')
        graph = nx.read_graphml(folder/'full.graphml')
        assert matrix.index.tolist() == matrix.columns.tolist() == nodes
        values = matrix.to_numpy()
        assert np.isfinite(values).all() and np.max(np.abs(values-values.T)) < 1e-12
        assert np.allclose(np.diag(values), 1, atol=1e-12, rtol=0)
        assert len(edges) == len(expected) and set(zip(edges.source, edges.target)) == expected
        assert set(graph.nodes) == set(nodes) and graph.number_of_edges() == len(expected)
        count = summary['valid_bootstraps']
        assert count + summary['invalid_bootstraps'] == 1000 and count >= 800
        for row in edges.itertuples():
            assert abs(row.weight-matrix.loc[row.source, row.target]) < 1e-12
            assert row.ci_lower <= row.ci_upper and -1.000000000001 <= row.ci_lower <= 1.000000000001
            assert -1.000000000001 <= row.ci_upper <= 1.000000000001
            frequency = row.top20_bootstrap_frequency
            assert 0 <= frequency <= 1 and abs(frequency*count-round(frequency*count)) < 1e-9
            for key in ['weight', 'ci_lower', 'ci_upper', 'top20_bootstrap_frequency']:
                assert abs(graph[row.source][row.target][key]-getattr(row, key)) < 1e-12
        assert abs(edges.top20_bootstrap_frequency.sum()-20) < 1e-9
        rho = float(spearmanr(values[upper], gold.to_numpy()[upper]).statistic)
        assert abs(rho-summary['point']['edge_weight_spearman']) < 1e-12
        checks[method] = {'nodes': len(nodes), 'edges': len(edges), 'valid_bootstraps': count,
                          'recomputed_edge_spearman': rho, 'graphml_attributes_match': True}
        sources.extend(folder/name for name in ['full-adjacency.csv', 'edges.csv', 'full.graphml'])
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'verification.json').write_text(json.dumps({'status': 'passed', 'methods': checks,
        'source_sha256': {str(p): file_hash(p) for p in sources}, 'script_sha256': file_hash(Path(__file__)),
        'scope': 'Export consistency, node order, bootstrap frequency accounting and matrix-to-human edge correlation; does not independently reconstruct bootstrap draws or establish construct validity.'}, indent=2))


if __name__ == '__main__':
    main()
