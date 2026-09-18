"""Show fixed-reference distance invariance and distinct within-material anchor ranks."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = {v: args.analysis/('base-'+v+'.npz') for v in ['raw_logp', 'reference_log_ratio', 'reference_z']}
    arrays = {v: np.load(p) for v, p in paths.items()}
    raw = arrays['raw_logp']
    for data in arrays.values():
        for key in ['sample_ids', 'anchor_ids', 'bridge_ids']:
            if not np.array_equal(data[key], raw[key]):
                raise ValueError('Unaligned matrices.')
    values = {v: data['cube'].mean(axis=2) for v, data in arrays.items()}
    distances = {v: pdist(matrix) for v, matrix in values.items()}
    error = float(np.max(np.abs(distances['raw_logp']-distances['reference_log_ratio'])))
    if error > 1e-9:
        raise ValueError('Distance invariance failed.')
    rows = []
    for i, identifier in enumerate(raw['sample_ids']):
        for variant in ['reference_log_ratio', 'reference_z']:
            rho = spearmanr(values['raw_logp'][i], values[variant][i]).statistic
            rows.append(dict(sample_id=identifier, comparison=variant, spearman=rho))
    table = pd.DataFrame(rows)
    table.to_csv(args.output/'within-material-ranks.csv', index=False)
    plt.style.use(['science', 'no-latex', 'bright'])
    plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    ax = axes[0]
    ax.scatter(distances['raw_logp'], distances['reference_log_ratio'], s=4, alpha=.2, rasterized=True)
    maximum = max(distances['raw_logp'])
    ax.plot([0, maximum], [0, maximum], color='.4', linestyle='--', linewidth=.7)
    ax.set(xlabel='Raw-score Euclidean distance', ylabel='Log-ratio Euclidean distance', title='A  Fixed-reference translation')
    ax.text(.04, .94, f'Maximum error: {error:.2e}', transform=ax.transAxes, va='top', fontsize=9)
    ax = axes[1]
    for variant, color, label in [('reference_log_ratio', '#4477AA', 'Raw vs. log-ratio'), ('reference_z', '#EE6677', 'Raw vs. z')]:
        observed = table.loc[table.comparison == variant, 'spearman'].dropna()
        ax.hist(observed, bins=np.linspace(-1, 1, 17), histtype='step', linewidth=1.6, color=color, label=label)
    ax.set(xlabel='Within-material anchor-rank Spearman correlation', ylabel='Number of materials', title='B  Anchor rankings may change', xlim=(-1.03, 1.03))
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    fig.subplots_adjust(left=.08, right=.985, bottom=.23, top=.86, wspace=.35)
    fig.text(.08, .04, '128 frozen English materials; three-bridge mean. All 8,128 material pairs retained.\nTranslation preserves between-material distances, not comparisons between different anchors within a material.', fontsize=8)
    for ext in ['pdf', 'png']:
        fig.savefig(args.output/f'calibration-final.{ext}', dpi=300)
    plt.close(fig)
    (args.output/'manifest.json').write_text(json.dumps({'max_distance_error': error,
        'undefined_ranks': int(table.spearman.isna().sum()), 'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.values()},
        'scope': 'Descriptive complete protected base-instrument matrices; no external labels or method selection'}, indent=2))


if __name__ == '__main__':
    main()
