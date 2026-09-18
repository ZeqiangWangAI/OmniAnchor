"""Resolve complete Slurm sensitivity instruments before the frozen CPU analysis."""
import argparse
from pathlib import Path
import subprocess
import sys

if __package__:
    from .run_analysis_request import completed_array_runs
else:
    from run_analysis_request import completed_array_runs
from omnianchor.io import read_json, write_json


def match_runs(rows, runs):
    by_spec = {row['spec_sha256']: row for row in rows}
    if len(by_spec) != len(rows):
        raise ValueError('Duplicate instrument specifications.')
    result = {}
    for run in runs:
        manifest = read_json(Path(run)/'measurement/manifest.json')
        row = by_spec.get(manifest['spec_sha256'])
        if row is None or row['variant'] in result:
            raise ValueError('Unknown or repeated sensitivity instrument.')
        result[row['variant']] = dict(reference_runs=row['reference_runs'], runs=[str(run)])
    if len(result) != len(rows):
        raise ValueError('Incomplete sensitivity instrument coverage.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['bank', 'prepared', 'runs-root', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--study', choices=['emobank', 'dwug'], required=True)
    args = parser.parse_args()
    rows = read_json(args.prepared/'shards.json')['rows']
    runs = completed_array_runs(args.job_id, len(rows), 'development', args.runs_root)
    mapping = args.output.parent/'sensitivity-resolved-runs.json'
    write_json(mapping, match_runs(rows, runs))
    subprocess.run([sys.executable, str(Path(__file__).with_name('analyze_sensitivity.py')),
        '--bank', str(args.bank), '--runs', str(mapping), '--study', args.study,
        '--frozen-evaluation', str(args.prepared/'analysis-contract.json'), '--output', str(args.output)], check=True)


if __name__ == '__main__':
    main()
