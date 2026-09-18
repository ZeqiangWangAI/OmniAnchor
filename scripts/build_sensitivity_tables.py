"""Export complete frozen sensitivity tables and readable supplementary summaries."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    events = (args.analysis/'events.jsonl').read_text().splitlines()
    if not events or json.loads(events[-1])['status'] != 'completed':
        raise ValueError('Analysis must complete before table generation.')
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.analysis/'manifest.json').read_text())
    clusters = json.loads((args.analysis/'clusters.json').read_text())
    geometry = pd.read_csv(args.analysis/'geometry-agreement.csv')
    lines = ['# Frozen instrument sensitivity', '',
        f"Population: {len(manifest['sample_ids'])} protected materials; {len(set(manifest['source_groups']))} source groups.",
        '', 'All prescribed instruments and scoring variants are retained. Intervals are descriptive, conditional on the frozen materials and reference. No external labels or method selection are used.',
        '', '| Instrument | Score variant | Distance-rank agreement | 95% interval |', '|---|---|---:|---|']
    def number(value):
        return 'undefined' if pd.isna(value) else f'{value:.3f}'
    for row in geometry.itertuples():
        lines.append(f'| {row.instrument} | {row.variant} | {number(row.estimate)} | [{number(row.lower)}, {number(row.upper)}] |')
    lines += ['', '## Descriptive clustering', '',
        'Fixed k=2, seed42, n_init10. Bootstrap refits predict the same original nodes. These are stability summaries, not human cluster validity.', '',
        '| Instrument | Template mean ARI | Template interval | Bootstrap mean ARI | Bootstrap interval | Valid draws |',
        '|---|---:|---|---:|---|---:|']
    for name, value in clusters.items():
        interval = lambda key: 'undefined' if value.get(key) is None else '['+', '.join(number(v) for v in value[key])+']'
        lines.append(f"| {name} | {number(value.get('template_mean_ari'))} | {interval('template_ari_interval')} | {number(value.get('fixed_node_bootstrap_ari_mean'))} | {interval('fixed_node_bootstrap_ari_interval')} | {value.get('valid_resamples', 0)} |")
    lines += ['', '## Complete machine-readable evidence', '',
        'The accompanying files retain every coordinate comparison, token-count record, invariant check and clustering field. Undefined results remain undefined; they are not set to zero.', '']
    names = ['geometry-agreement.csv', 'coordinate-agreement.csv', 'token-counts.csv', 'invariants.json', 'clusters.json']
    hashes = {}
    for name in names:
        source = args.analysis/name
        shutil.copyfile(source, args.output/name)
        hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        lines.append(f'- [{name}]({name})')
    (args.output/'sensitivity.md').write_text('\n'.join(lines)+'\n')
    (args.output/'manifest.json').write_text(json.dumps({'source_sha256': hashes,
        'analysis_manifest_sha256': hashlib.sha256((args.analysis/'manifest.json').read_bytes()).hexdigest(),
        'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2))


if __name__ == '__main__':
    main()
