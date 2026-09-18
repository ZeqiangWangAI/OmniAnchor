"""Independently recompute VIVA AP/MRR from raw candidate scores without evaluator APIs."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vlanchor.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['analysis', 'raw-root', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.analysis/'manifest.json').read_text())
    for name, digest in manifest['input_sha256'].items():
        assert file_hash(Path(name)) == digest
    candidates_path = next(Path(p) for p in manifest['input_sha256'] if p.endswith('candidates.json'))
    labels_path = next(Path(p) for p in manifest['input_sha256'] if p.endswith('labels.csv'))
    candidates = json.loads(candidates_path.read_text())
    labels = pd.read_csv(labels_path).set_index(['sample_id', 'anchor_id']).relevant
    expected = pd.read_csv(args.analysis/'per-recipient.csv').set_index(['method', 'condition', 'sample_id', 'metric']).value
    runs = {'native': 'viva-development-46833', 'qwen-embedding': 'viva-development-47086', 'qwen-reranker': 'viva-development-45850'}
    checked, hashes = [], {}
    for method, run in runs.items():
        for condition in ['image_action', 'action_only', 'mismatched_image_action']:
            folder = args.raw_root/run/method/condition
            seen = set()
            for path in sorted(folder.glob('part-*')):
                if path.suffix not in {'.parquet', '.npz'}:
                    continue
                hashes[str(path)] = file_hash(path)
                if method == 'native':
                    frame = pd.read_parquet(path)
                    assert frame.sample_id.nunique() == 1 and set(frame.status) == {'ok'}
                    assert not frame.duplicated(['anchor_id', 'bridge_id']).any()
                    assert frame.groupby('anchor_id').bridge_id.nunique().nunique() == 1
                    identifier = frame.sample_id.iloc[0]
                    scores = frame.groupby('anchor_id').raw_logp.mean().to_dict()
                else:
                    with np.load(path, allow_pickle=False) as data:
                        identifier = str(data['sample_id'])
                        anchors = data['anchor_ids'].tolist()
                        assert len(anchors) == len(set(anchors))
                        scores = dict(zip(anchors, data['scores']))
                assert identifier not in seen
                seen.add(identifier)
                order = [c['id'] for c in candidates[identifier]]
                assert set(scores) == set(order)
                y = np.array([labels.loc[(identifier, a)] for a in order], dtype=int)
                s = np.array([scores[a] for a in order], dtype=float)
                assert np.isfinite(s).all() and set(y) == {0, 1}
                ranked = np.argsort(-s, kind='stable')
                ys, ss = y[ranked], s[ranked]
                ends = np.r_[np.flatnonzero(np.diff(ss) != 0), len(ss)-1]
                tp = np.cumsum(ys)[ends]
                ap = float(np.sum(np.diff(np.r_[0, tp])/y.sum() * tp/(ends+1)))
                rr = float(1/(np.flatnonzero(ys)[0]+1))
                for metric, actual in [('AP', ap), ('MRR', rr)]:
                    error = abs(actual-expected.loc[(method, condition, identifier, metric)])
                    assert error < 1e-12
                    checked.append({'method': method, 'condition': condition, 'sample_id': identifier, 'metric': metric, 'value': actual, 'error': error})
            assert seen == set(manifest['samples'])
    assert len(checked) == len(expected) == 3888
    args.output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(checked).to_csv(args.output/'verified-metrics.csv', index=False)
    (args.output/'manifest.json').write_text(json.dumps({'status': 'passed', 'comparisons': len(checked),
        'max_error': max(r['error'] for r in checked), 'source_sha256': hashes,
        'script_sha256': file_hash(Path(__file__)), 'scope': 'Raw bridge means, candidate/label alignment, threshold-tie AP and stable-order MRR; bootstrap intervals not independently recomputed.'}, indent=2))


if __name__ == '__main__':
    main()
