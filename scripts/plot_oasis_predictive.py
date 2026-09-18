"""Plot all six frozen OASIS predictors, keeping direct measurement separate."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots  # noqa: F401

from omnianchor.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = pd.read_csv(args.metrics)
    methods = ['native-raw_logp', 'native-reference_log_ratio', 'native-reference_z',
               'qwen-embedding-anchor', 'qwen-embedding-original', 'qwen-reranker-anchor']
    labels = ['OmniAnchor: raw', 'OmniAnchor: log ratio', 'OmniAnchor: reference z',
              'Embedding: anchors', 'Embedding: original', 'Reranker: anchors']
    expected = {(m, c) for m in methods for c in ['spearman_V', 'spearman_A']}
    if (len(data) != 12 or set(zip(data.method, data.metric)) != expected
            or set(data.n) != {201} or set(data.kind) != {'probe'}):
        raise ValueError('Require all six frozen predictors on 201 images.')
    args.output.mkdir(parents=True, exist_ok=False)
    plt.style.use(['science', 'no-latex', 'bright'])
    plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(9, 5.2), sharey=True)
    colors = ['#4477AA']*3+['#228833']*2+['#EE6677']
    for ax, metric, title in zip(axes, ['spearman_V', 'spearman_A'], ['A  Valence', 'B  Arousal']):
        subset = data[data.metric == metric].set_index('method')
        for y, (method, color) in enumerate(zip(methods, colors)):
            row = subset.loc[method]
            ax.errorbar(row.value, y, xerr=[[row.value-row.ci_lower], [row.ci_upper-row.value]],
                        fmt='o', color=color, capsize=3, markersize=5)
        ax.set_title(title, loc='left', fontsize=12)
        ax.set(xlim=(.25, .95), xlabel='Spearman correlation', ylim=(5.5, -.5))
        ax.set_xticks([.3, .5, .7, .9])
        ax.grid(axis='x', color='.9', linewidth=.6)
    axes[0].set_yticks(range(6), labels)
    fig.subplots_adjust(left=.23, right=.975, bottom=.31, top=.90, wspace=.18)
    fig.text(.23, .075, '201 protected images; fixed train-fitted Ridge predictors.\n95% image-bootstrap intervals; marginal intervals are not paired tests.\nPredictive performance is distinct from direct measurement validity.', fontsize=9)
    for suffix in ['pdf', 'png']:
        fig.savefig(args.output/('oasis12-predictive.'+suffix), dpi=300)
    plt.close(fig)
    data.to_csv(args.output/'plotted-results.csv', index=False)
    (args.output/'manifest.json').write_text(json.dumps({'source_sha256': {str(args.metrics): file_hash(args.metrics)},
        'script_sha256': file_hash(Path(__file__)), 'n_plotted_results': 12}, indent=2))


if __name__ == '__main__':
    main()
