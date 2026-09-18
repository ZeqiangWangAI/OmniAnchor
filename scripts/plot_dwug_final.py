"""Plot frozen DWUG final results without refitting or selecting target words."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import scienceplots  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    shift = pd.read_csv(args.analysis/'per-target-shift.csv')
    native = shift[shift.method == 'native-raw_logp']
    pairs = json.loads((args.analysis/'pooled-pair-validity.json').read_text())
    methods = ['native-raw_logp', 'e5-original', 'qwen-embedding-original', 'qwen-reranker-anchor']
    labels = ['OmniAnchor', 'E5 original', 'Qwen embedding', 'Qwen reranker']
    plt.style.use(['science', 'no-latex', 'bright'])
    plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={'width_ratios': [1.05, 1]})
    ax = axes[0]
    ax.scatter(native.human_change, native.energy_distance, s=36, color='#4477AA', zorder=3)
    offsets = {'include_vb': (-35, 10), 'face_nn': (-14, 14), 'part_nn': (-18, -22), 'land_nn': (12, -22),
               'multitude_nn': (-12, 24), 'grain_nn': (16, 10)}
    for row in native.itertuples():
        ax.annotate(row.target.rsplit('_', 1)[0], (row.human_change, row.energy_distance),
                    xytext=offsets.get(row.target, (5, 5)), textcoords='offset points', fontsize=8,
                    arrowprops={'arrowstyle': '-', 'color': '.65', 'lw': .5})
    ax.margins(x=.24, y=.25)
    ax.set(xlabel='Human semantic-change score', ylabel='Native energy distance',
           title='A  Change ranking: 9 held-out words')
    ax.text(.04, .96, r'$\rho = .200$; 95% CI [−.684, .893]', transform=ax.transAxes, va='top', fontsize=9)
    ax = axes[1]
    for i, method in enumerate(methods):
        d = pairs[method]['target_group_bootstrap']; x = d['estimate']
        ax.errorbar(x, i, xerr=[[x-d['lower']], [d['upper']-x]], fmt='o', capsize=3,
                    color='#4477AA' if i == 0 else '#888888')
    ax.set(yticks=range(4), yticklabels=labels, xlabel='Spearman correlation with human relatedness',
           title='B  Usage-pair validity: 10,120 pairs', xlim=(-.03, .72))
    ax.invert_yaxis(); ax.axvline(0, color='.7', linewidth=.7)
    ax.text(.02, -.24, '95% intervals resample 9 whole target groups.\nShared sources and annotators limit independence.',
            transform=ax.transAxes, fontsize=8)
    fig.subplots_adjust(left=.075, right=.98, bottom=.25, top=.88, wspace=.56)
    for ext in ['pdf', 'png']:
        fig.savefig(args.output/f'dwug-final.{ext}', dpi=300)
    plt.close(fig)
    sources = [args.analysis/'per-target-shift.csv', args.analysis/'pooled-pair-validity.json', Path(__file__)]
    (args.output/'manifest.json').write_text(json.dumps({'scope': 'All nine final targets; four prespecified raw/original method displays; complete eight-method results retained in analysis',
        'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}, indent=2))


if __name__ == '__main__':
    main()
