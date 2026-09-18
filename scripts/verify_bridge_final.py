"""Verify every saved bridge AP with the existing independent base-R implementation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['analysis', 'labels', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.analysis/'manifest.json').read_text())
    if hashlib.sha256(args.labels.read_bytes()).hexdigest() != manifest['labels_sha256']:
        raise ValueError('Original frozen labels changed.')
    args.output.mkdir(parents=True, exist_ok=False)
    labels = pd.read_csv(args.labels)
    columns = labels.columns.drop('sample_id').tolist()
    if labels.sample_id.duplicated().any() or set(labels.sample_id) != set(manifest['sample_ids']):
        raise ValueError('Protected label population differs.')
    (args.output/'column-ids.txt').write_text('\n'.join(columns)+'\n')
    labels[['sample_id']].to_csv(args.output/'labels.index.csv', index=False)
    labels[columns].to_numpy(dtype='<f8').tofile(args.output/'labels.f64')
    metrics = json.loads((args.analysis/'metrics.json').read_text())
    rows, hashes = [], {}
    for key, metric in metrics.items():
        name = key.replace('/', '-')
        path = args.analysis/(name+'.npz')
        saved = np.load(path)
        ids, anchors = saved['sample_ids'].tolist(), saved['anchor_ids'].tolist()
        if len(set(ids)) != len(ids) or set(ids) != set(labels.sample_id) or set(anchors) != set(columns):
            raise ValueError('Score population or anchors differ.')
        values = saved['values'][:, [anchors.index(c) for c in columns]]
        if not np.isfinite(values).all():
            raise ValueError('Invalid scores.')
        values.astype('<f8').tofile(args.output/('direct--'+name+'.f64'))
        pd.DataFrame({'sample_id': ids}).to_csv(args.output/('direct--'+name+'.index.csv'), index=False)
        rows.append(dict(kind='direct', method=name, metric='macro_ap', value=metric['multilabel']['macro_ap']))
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    pd.DataFrame(rows).to_csv(args.output/'reported.csv', index=False)
    rscript = Path(__file__).with_name('verify_multilabel_final.R')
    subprocess.run(['Rscript', str(rscript), str(args.output), str(args.labels),
                    str(args.output/'reported.csv'), 'binary64'], check=True)
    result = pd.read_csv(args.output/'verification.csv')
    if len(result) != len(metrics) or len(result) != 18 or result.max_error.max() >= 1e-12:
        raise ValueError('Incomplete independent verification.')
    (args.output/'manifest.json').write_text(json.dumps({'status': 'passed', 'n_metrics': len(result),
        'max_error': result.max_error.max(), 'input_sha256': hashes,
        'labels_sha256': manifest['labels_sha256'], 'R_sha256': hashlib.sha256(rscript.read_bytes()).hexdigest(),
        'scope': 'All 18 saved AP outcomes with exact binary64 score transfer; no probe refit or independent bootstrap verification'}, indent=2))


if __name__ == '__main__':
    main()
