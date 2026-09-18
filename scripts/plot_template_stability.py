"""Plot frozen descriptive clustering stability with distinct bootstrap estimands."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--analysis', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
source = args.analysis/'clusters.json'
data = json.loads(source.read_text())
args.output.mkdir(parents=True, exist_ok=False)
plt.style.use(['science', 'no-latex', 'bright'])
plt.rcParams.update({'font.family': 'STIXGeneral', 'font.size': 10})
fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharey=True)
for ax, point, interval, title, color in zip(axes,
        ['template_mean_ari', 'fixed_node_bootstrap_ari_mean'],
        ['template_ari_interval', 'fixed_node_bootstrap_ari_interval'],
        ['A  Agreement across templates', 'B  Source-resampled cluster fits'],
        ['#4477AA', '#228833']):
    for i, (name, row) in enumerate(data.items()):
        if row['status'] != 'ok' or row.get(interval) is None:
            ax.text(0, i, 'Undefined', va='center')
            continue
        low, high = row[interval]
        ax.plot([low, high], [i, i], color=color, linewidth=2)
        ax.plot(row[point], i, 'o', color=color)
    ax.axvline(0, color='.7', linewidth=.7, linestyle='--')
    ax.set(xlim=(-.1, 1.02), ylim=(len(data)-.4, -.6), xlabel='Adjusted Rand index', title=title)
    ax.set_yticks(range(len(data)), [n.replace('-', ' ').capitalize() for n in data])
fig.subplots_adjust(left=.15, right=.98, top=.85, bottom=.24, wspace=.15)
fig.text(.15, .065, 'Fixed k = 2; seed = 42; 1,000 source-group bootstrap draws. Bars: 95% intervals.\nSame original nodes in each comparison. Descriptive stability does not establish human cluster validity.', fontsize=8)
for ext in ['png', 'pdf']:
    fig.savefig(args.output/f'template-stability.{ext}', dpi=300)
plt.close(fig)
(args.output/'manifest.json').write_text(json.dumps({'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'plotted': data, 'scope': 'Saved point estimates and intervals; no new inference.'}, indent=2))
