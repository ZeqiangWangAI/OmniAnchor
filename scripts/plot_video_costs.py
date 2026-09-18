"""Plot observed small-sample video capability costs without extrapolation."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots  # noqa: F401

from vlanchor.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--costs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = pd.read_csv(args.costs)
    methods = ['native', 'qwen-embedding', 'qwen-reranker']
    assert len(data) == 6 and set(zip(data.method, data.frames)) == {(m, f) for m in methods for f in [8, 16]}
    assert set(data.mixed_inputs) == {32} and set(data.videos) == {16} and set(data.texts) == {16}
    args.output.mkdir(parents=True, exist_ok=False)
    plt.style.use(['science', 'no-latex', 'bright'])
    plt.rcParams.update({'font.family': 'STIXGeneral', 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.8))
    for method, label, color, marker in zip(methods, ['VLanchor', 'Official embedding', 'Official reranker'],
                                            ['#4477AA', '#228833', '#EE6677'], ['o', 's', '^']):
        rows = data[data.method == method].sort_values('frames')
        for ax, values in zip(axes, [rows.measurement_seconds/60, rows.peak_allocated_bytes/2**30]):
            ax.plot(rows.frames, values, marker=marker, color=color, label=label)
    for ax, title, ylabel in zip(axes, ['A  Measurement time', 'B  Peak allocated memory'],
                                  ['Minutes per mixed batch', 'GiB']):
        ax.set_title(title, loc='left', fontsize=12)
        ax.set(xlabel='Requested frames per video', ylabel=ylabel, xticks=[8, 16], xlim=(7, 17))
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', color='.9', linewidth=.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.53, .16), ncol=3, frameon=False)
    fig.subplots_adjust(left=.10, right=.98, bottom=.36, top=.89, wspace=.32)
    fig.text(.10, .045, 'Each observation: 16 video + 16 text inputs. One completed batch per method/frame setting.\nNo uncertainty interval or per-video latency claim. One-frame failures are retained separately.', fontsize=9)
    for ext in ['pdf', 'png']:
        fig.savefig(args.output/('video-costs.'+ext), dpi=300)
    plt.close(fig)
    data.to_csv(args.output/'plotted-costs.csv', index=False)
    (args.output/'manifest.json').write_text(json.dumps({'source_sha256': {str(args.costs): file_hash(args.costs)},
        'script_sha256': file_hash(Path(__file__)), 'unit_conversions': 'seconds/60; allocated bytes/2**30',
        'scope': 'Six observed mixed-batch costs; no inference or scaling extrapolation.'}, indent=2))


if __name__ == '__main__':
    main()
