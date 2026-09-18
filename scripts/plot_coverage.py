"""Render verified study coverage; filled cells indicate execution, not validity success."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401


def family(method):
    if method.startswith('native') or method == 'OmniAnchor' or method.startswith('qwen35-'):
        return 'Qwen3.5-4B'
    if method.startswith('qwen3vl-'):
        return 'Qwen3-VL-4B'
    if method.startswith('e5-'):
        return 'E5'
    if method.startswith('qwen-embedding'):
        return 'Qwen embedding'
    if method.startswith('qwen-reranker'):
        return 'Qwen reranker'
    if method.startswith('bert-'):
        return 'BERT'
    if method == 'roberta-base':
        return 'RoBERTa'
    return 'Other FMAT models'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    data = pd.read_csv(args.ledger)
    columns = ['Qwen3.5-4B', 'Qwen3-VL-4B', 'E5', 'Qwen embedding', 'Qwen reranker', 'BERT', 'RoBERTa', 'Other FMAT models']
    studies = ['FMAT d1a', 'FMAT d1b', 'ValueEval', 'EmoBank', 'Chinese EmoBank', 'OASIS direct', 'OASIS predictive', 'VIVA', 'DWUG', 'Video demo 8 frames', 'Video demo 16 frames']
    if set(data.study) != set(studies):
        raise ValueError('Unrepresented study in coverage ledger.')
    matrix = np.zeros((len(studies), len(columns)), dtype=int)
    labels = []
    for i, study in enumerate(studies):
        rows = data[data.study == study]
        first = rows.iloc[0]
        labels.append(f'{study} | {first.modality} | {first.language}')
        for row in rows.itertuples():
            j = columns.index(family(row.method))
            matrix[i, j] = 2 if row.status.startswith('capability') else 1
    plt.style.use(['science', 'no-latex', 'bright'])
    plt.rcParams.update({'font.family': 'STIXGeneral', 'font.size': 9})
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.imshow(matrix, cmap=ListedColormap(['#F3F3F3', '#4477AA', '#CCBB44']), vmin=0, vmax=2, aspect='auto')
    ax.set_xticks(range(len(columns)), columns, rotation=35, ha='right')
    ax.set_yticks(range(len(studies)), labels)
    for i in range(len(studies)):
        for j in range(len(columns)):
            if matrix[i, j]:
                ax.text(j, i, 'D' if matrix[i, j] == 2 else 'E', ha='center', va='center', color='white' if matrix[i, j] == 1 else '#333333', fontsize=9)
    ax.set_xticks(np.arange(-.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(studies), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=2)
    ax.tick_params(which='both', length=0, top=False, right=False)
    ax.set_title('Observed study coverage: completed outcomes and capability demonstrations', fontsize=12, pad=12)
    fig.subplots_adjust(left=.44, right=.985, top=.89, bottom=.29)
    fig.legend(handles=[Patch(color='#4477AA', label='E: completed evaluation; positive or negative'), Patch(color='#CCBB44', label='D: capability demonstration'), Patch(color='#F3F3F3', label='Not established in this ledger')], loc='lower center', bbox_to_anchor=(.55, .065), ncol=3, frameon=False, fontsize=8)
    fig.text(.04, .018, 'Model-family grouping retains all ledger methods. E3 bounded sensitivity is reported separately.\nColored coverage does not imply superiority, cross-language equivalence, or successful construct validation.', fontsize=8)
    for ext in ['pdf', 'png']:
        fig.savefig(args.output/f'coverage.{ext}', dpi=300)
    plt.close(fig)
    (args.output/'manifest.json').write_text(json.dumps({'ledger_sha256': hashlib.sha256(args.ledger.read_bytes()).hexdigest(), 'studies': studies, 'model_families': columns, 'matrix': matrix.tolist()}, indent=2))


if __name__ == '__main__':
    main()
