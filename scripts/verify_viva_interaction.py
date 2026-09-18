"""Recompute recipient-level interaction summaries from saved coordinate deltas."""
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
    source = args.analysis/'per-coordinate-interactions.csv'
    data = pd.read_csv(source, float_precision='round_trip')
    expected = pd.read_csv(args.analysis/'interaction-summary.csv')
    coverage = json.loads((args.analysis/'coverage.json').read_text())
    if data.duplicated(['method', 'sample_id', 'anchor_id']).any() or not np.isfinite(data.delta).all():
        raise ValueError('Duplicate or invalid coordinates.')
    if set(data.method) != set(expected.method) or len(expected) != 2 * data.method.nunique():
        raise ValueError('Method coverage mismatch.')
    rows = []
    for method, frame in data.groupby('method'):
        grouped = frame.groupby('sample_id').delta
        if grouped.ngroups != coverage['recipients']:
            raise ValueError('Recipient coverage mismatch.')
        signed = grouped.mean().mean()
        rms = grouped.apply(lambda values: np.sqrt(np.square(values).mean())).mean()
        for metric, actual in [('mean_signed_delta', signed), ('mean_recipient_rms_delta', rms)]:
            selected = expected[(expected.method == method) & (expected.metric == metric)]
            if len(selected) != 1:
                raise ValueError('Missing or duplicate summary.')
            error = abs(actual-selected.estimate.iloc[0])
            if error >= 1e-12:
                raise ValueError('Summary mismatch.')
            rows.append({'method': method, 'metric': metric, 'actual': actual, 'absolute_error': error})
    pd.DataFrame(rows).to_csv(args.output/'verified-means.csv', index=False)
    (args.output/'manifest.json').write_text(json.dumps({'status': 'passed', 'n_summaries': len(rows),
        'scope': 'Recipient-weighted means recomputed from saved deltas; does not independently verify raw four-condition scoring or bootstrap intervals',
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}, indent=2))


if __name__ == '__main__':
    main()
