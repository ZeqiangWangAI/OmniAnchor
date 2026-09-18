"""Render all frozen OASIS affect12 predictor outcomes and no-content contrasts."""
import argparse
import json
from pathlib import Path

import pandas as pd

from build_verified_paper_tables import emit_table
from vlanchor.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['analysis', 'controls', 'verification', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    for folder in [args.analysis, args.controls, args.verification]:
        if json.loads((folder/'events.jsonl').read_text().splitlines()[-1])['status'] != 'completed':
            raise ValueError('Require completed analysis, controls and independent verification.')
    metrics = pd.read_csv(args.analysis/'metrics.csv')
    paired = pd.read_csv(args.analysis/'paired-differences.csv')
    controls = pd.read_csv(args.controls/'paired-control-differences.csv')
    if (len(metrics) != 12 or metrics.method.nunique() != 6 or set(metrics.n) != {201}
            or set(metrics.kind) != {'probe'} or len(paired) != 4 or len(controls) != 6):
        raise ValueError('Unexpected frozen study coverage.')
    args.output.mkdir(parents=True, exist_ok=False)
    def interval(row):
        return f'{row.value:.3f} [{row.ci_lower:.3f}, {row.ci_upper:.3f}]'
    rows = []
    for method in metrics.method.drop_duplicates():
        selected = metrics[metrics.method == method].set_index('metric')
        rows.append([method, *[interval(selected.loc['spearman_'+c]) for c in ['V', 'A']]])
    emit_table(args.output, 'oasis12-predictive',
        'OASIS affect12: frozen train-fitted Ridge predictors on 201 protected images. Spearman correlation [95% image-bootstrap CI]. These are predictive results, not direct measurement validity.',
        ['Features', 'Valence', 'Arousal'], rows)
    for name, frame in [('metrics', metrics), ('paired-differences', paired), ('no-content-paired', controls)]:
        frame.to_csv(args.output/(name+'.csv'), index=False)
    sources = [args.analysis/'metrics.csv', args.analysis/'paired-differences.csv',
               args.controls/'paired-control-differences.csv', args.verification/'verification.csv']
    (args.output/'manifest.json').write_text(json.dumps({
        'source_sha256': {str(p): file_hash(p) for p in sources},
        'script_sha256': file_hash(Path(__file__)), 'n_images': 201, 'n_methods': 6,
        'scope': 'Frozen predictors; four paired correlation comparisons and six no-content MSE contrasts are separate BH families.'}, indent=2))


if __name__ == '__main__':
    main()
