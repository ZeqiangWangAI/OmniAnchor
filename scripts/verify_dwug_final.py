"""Export exact saved DWUG predictions for independent base-R rank verification."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    gold = pd.read_csv(args.analysis/'aggregated-pair-gold.csv')
    predictions = np.load(args.analysis/'pair-predictions.npz')
    expected = json.loads((args.analysis/'pooled-pair-validity.json').read_text())
    rows = []
    for method in predictions.files:
        values = predictions[method]
        if values.shape != (len(gold),) or not np.isfinite(values).all():
            raise ValueError('Invalid saved pair coverage.')
        values.astype('<f8').tofile(args.output/(method+'.bin'))
        rows.append({'method': method, 'spearman': expected[method]['spearman']})
    gold.judgment.to_numpy(dtype='<f8').tofile(args.output/'gold.bin')
    pd.DataFrame(rows).to_csv(args.output/'expected.csv', index=False)
    script = Path(__file__).with_suffix('.R')
    subprocess.run(['Rscript', str(script), str(args.analysis), str(args.output), str(len(gold))], check=True)
    sources = [args.analysis/name for name in ['aggregated-pair-gold.csv', 'pair-predictions.npz',
        'pooled-pair-validity.json', 'per-target-shift.csv', 'change-validity.csv']]
    sources += [Path(__file__), script]
    (args.output/'manifest.json').write_text(json.dumps({'status': 'passed',
        'scope': 'Independent R recomputation of 8 pooled pair and 32 across-target rank correlations; does not independently validate input feature extraction or confidence intervals',
        'n_pairs': len(gold), 'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}, indent=2))


if __name__ == '__main__':
    main()
