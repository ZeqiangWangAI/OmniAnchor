"""Display all frozen bridge ablations, preserving the primary/secondary distinction."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.analysis/'paired-differences.csv'
    data = pd.read_csv(source)
    metrics = json.loads((args.analysis/'metrics.json').read_text())
    names = ['reference_guided', 'validity', 'reliability', 'alpha', 'random']
    labels = ['Reference-guided', 'Validity only', 'Reliability only', 'Alpha only', 'Random']
    plt.style.use(['science', 'no-latex', 'bright'])
    plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9), sharey=True)
    for ax, variant, title in zip(axes, ['reference_z', 'raw_logp'],
            ['A  Primary: reference-z', 'B  Secondary: raw scores']):
        rows = data[data.variant == variant].set_index('method').loc[names]
        if len(rows) != 5 or rows.index.has_duplicates:
            raise ValueError('Missing or repeated ablation.')
        for i, (name, row) in enumerate(rows.iterrows()):
            actual = metrics[name+'/'+variant]['multilabel']['macro_ap'] - metrics['fixed3/'+variant]['multilabel']['macro_ap']
            if not np.isclose(actual, row.difference, rtol=0, atol=1e-12):
                raise ValueError('Difference does not reproduce reported AP.')
            ax.errorbar(row.difference, i, xerr=[[row.difference-row.ci_lower], [row.ci_upper-row.difference]],
                        fmt='o', capsize=3, color='#4477AA')
            ax.text(.0108, i, f'{row.bh_q:.3f}', va='center', fontsize=9)
        ax.axvline(0, color='.55', linewidth=.8, linestyle='--')
        ax.set(xlim=(-.020, .016), ylim=(4.6, -.65), title=title,
               xlabel='Macro AP difference from fixed bridges', xticks=[-.02, -.01, 0, .01])
        ax.text(.0108, -.48, 'BH q', fontsize=9)
        ax.set_yticks(range(5), labels)
    fig.subplots_adjust(left=.17, right=.985, bottom=.25, top=.85, wspace=.17)
    fig.text(.17, .045, '1,576 held-out materials. Paired 95% source-group bootstrap intervals.\nCoinciding selected sets remain separate prescribed ablations; no independent-replication claim.', fontsize=8)
    for ext in ['pdf', 'png']:
        fig.savefig(args.output/f'bridge-final.{ext}', dpi=300)
    plt.close(fig)
    (args.output/'manifest.json').write_text(json.dumps({'verification': 'All ten plotted AP differences recomputed from saved method metrics within 1e-12; intervals and q copied from frozen analysis',
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'metrics_sha256': hashlib.sha256((args.analysis/'metrics.json').read_bytes()).hexdigest()}, indent=2))


if __name__ == '__main__':
    main()
