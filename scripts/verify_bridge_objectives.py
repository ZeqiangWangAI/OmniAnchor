"""Recompute saved selected-subset objectives without the optimization API."""
import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
labels_path = root / 'data/prepared/valueeval-20260910-01/search/labels.csv'
labels = pd.read_csv(labels_path, index_col='sample_id')
sources = {str(labels_path.relative_to(root)): hashlib.sha256(labels_path.read_bytes()).hexdigest()}
rows = []
for path in sorted(root.glob('runs/bridge-search-*/search/selected.json')):
    selected = json.loads(path.read_text())
    if selected['status'] != 'ok':
        continue
    sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    config = selected['manifest']['config']
    for trace in selected['trace']:
        if 'selected' not in trace:
            continue
        arrays = []
        for bridge in trace['selected']:
            p = path.parent / 'candidates' / bridge / 'search-z.npz'
            sources[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
            with np.load(p) as data:
                ids, anchors = data['sample_ids'].tolist(), data['anchor_ids'].tolist()
                arrays.append(pd.DataFrame(data['values'], index=ids, columns=anchors))
        assert all(a.index.equals(arrays[0].index) and a.columns.equals(arrays[0].columns) for a in arrays)
        y = labels.loc[arrays[0].index, arrays[0].columns].to_numpy()
        cube = np.stack([a.to_numpy() for a in arrays], axis=1)
        eligible = (y.sum(0) >= config['min_positive']) & ((1-y).sum(0) >= config['min_negative'])
        aps, correlations, alphas = [], [], []
        for j in np.flatnonzero(eligible):
            scores = cube[:, :, j].mean(1)
            # Threshold grouping preserves the average-precision treatment of ties.
            grouped = pd.DataFrame({'score': scores, 'positive': y[:, j]}).groupby('score')['positive'].agg(['sum', 'count']).sort_index(ascending=False)
            aps.append(float(((grouped['sum'].cumsum()/grouped['count'].cumsum())*grouped['sum']).sum()/y[:, j].sum()))
            ranks = pd.DataFrame(cube[:, :, j]).rank().to_numpy()
            correlations.append(np.mean([np.corrcoef(ranks[:, a], ranks[:, b])[0, 1] for a, b in combinations(range(cube.shape[1]), 2)]))
            items = cube[:, :, j]
            k = items.shape[1]
            alphas.append(k/(k-1)*(1-items.var(0, ddof=1).sum()/items.sum(1).var(ddof=1)))
        validity = float(np.mean(aps))
        reliability = float((1+np.median(correlations))/2)
        mode = selected['manifest']['mode']
        objective = {'reference_guided': config['validity_weight']*validity+(1-config['validity_weight'])*reliability, 'validity': validity, 'reliability': reliability, 'alpha': float(np.median(alphas)), 'random': 0.0}[mode]
        computed = {'validity_macro_ap': validity, 'reliability': reliability, 'objective': objective}
        errors = {key: abs(value-trace[key]) for key, value in computed.items()}
        assert max(errors.values()) < 1e-12, (path, trace['generation'], errors)
        rows.append({'run': path.parts[-3], 'generation': trace['generation'], 'mode': mode, 'computed': computed, 'absolute_errors': errors})
with args.output.open('x') as handle:
    json.dump({'status': 'passed', 'scope': 'Recomputes all saved selected-subset validity, reliability and selection objectives from training matrices. Does not certify unselected-subset optimality, generation RNG or independent raw-to-calibration reconstruction.', 'checks': rows, 'source_sha256': sources}, handle, indent=2)
print(f'Passed {len(rows)} selected-subset objective checks.')
