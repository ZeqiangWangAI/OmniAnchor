"""Plot frozen text validity, separating direct measurements from trained predictors."""
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
    for name in ['valueeval', 'english', 'chinese', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [args.valueeval, args.english, args.chinese]
    value, english, chinese = [pd.read_csv(p) for p in paths]
    plt.style.use(['science', 'no-latex', 'bright'])
    plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 10})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    labels = ['OmniAnchor', 'E5', 'Qwen embedding', 'Qwen reranker']
    colors = ['#4477AA', '#228833', '#CCBB44', '#EE6677']
    original = ['native-raw_logp', 'e5-original', 'qwen-embedding-original', 'qwen-reranker-anchor']
    anchor = ['native-raw_logp', 'e5-anchor', 'qwen-embedding-anchor', 'qwen-reranker-anchor']
    plotted = []
    for ax, data, kind, metrics, methods, title in [
        (axes[0, 0], value, 'direct', ['macro_ap'], anchor, 'A  ValueEval: direct measurements'),
        (axes[0, 1], value, 'probe', ['macro_ap'], original, 'B  ValueEval: trained predictors'),
        (axes[1, 0], english, 'probe', ['spearman_V', 'spearman_A', 'spearman_D'], original, 'C  English affect: trained predictors'),
        (axes[1, 1], chinese, 'probe', ['spearman_V', 'spearman_A'], original, 'D  Chinese affect: trained predictors')]:
        for j, (method, color, label) in enumerate(zip(methods, colors, labels)):
            for i, metric in enumerate(metrics):
                selected = data[(data.kind == kind) & (data.method == method) & (data.metric == metric)]
                if len(selected) != 1:
                    raise ValueError('Missing or repeated plotted result.')
                row = selected.iloc[0]
                x = j if len(metrics) == 1 else i+(j-1.5)*.16
                ax.errorbar(x, row.value, yerr=[[row.value-row.ci_lower], [row.ci_upper-row.value]],
                            fmt='o', color=color, capsize=3, label=label if i == 0 else None)
                plotted.append(dict(panel=title, **row.to_dict()))
        ax.set_title(title, loc='left', fontsize=11)
        if len(metrics) == 1:
            ax.set_xticks(range(4), labels, rotation=15, ha='right')
            ax.set(ylabel='Macro average precision', ylim=(.10, .50), xlim=(-.5, 3.5))
        else:
            ax.set_xticks(range(len(metrics)), [m[-1] for m in metrics])
            ax.set(ylabel='Spearman correlation', ylim=(.05, .9), xlim=(-.5, len(metrics)-.5))
        ax.grid(axis='y', color='.9', linewidth=.6)
    handles, names = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, names, ncol=4, loc='lower center', bbox_to_anchor=(.53, .05), frameon=False, fontsize=9)
    fig.subplots_adjust(left=.075, right=.985, bottom=.19, top=.94, hspace=.52, wspace=.27)
    fig.text(.075, .018, '95% source-group bootstrap intervals; marginal intervals are not paired superiority tests.\nDirect panels use anchor scores. Predictor panels use native coordinates, original E5/Qwen vectors, and reranker anchor scores.', fontsize=8)
    for ext in ['pdf', 'png']:
        fig.savefig(args.output/f'text-final.{ext}', dpi=300)
    plt.close(fig)
    pd.DataFrame(plotted).to_csv(args.output/'plotted-results.csv', index=False)
    (args.output/'manifest.json').write_text(json.dumps({'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'scope': 'Prespecified native raw and official baseline displays; all other frozen variants retained in complete tables',
        'n_plotted_results': len(plotted)}, indent=2))


if __name__ == '__main__':
    main()
