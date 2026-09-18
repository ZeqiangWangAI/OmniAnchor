"""Index unsuccessful campaign allocations without inferring failure causes."""
import argparse
import json
from pathlib import Path

import pandas as pd

from vlanchor.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accounting', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.accounting.read_text())
    assert len({r['JobIDRaw'] for r in data}) == len(data)
    rows = []
    for record in data:
        assert record['JobIDRaw'].isdigit()
        if record['State'].split()[0] in {'COMPLETED', 'RUNNING', 'PENDING', 'CONFIGURING', 'COMPLETING'}:
            continue
        rows.append({'job_id': record['JobIDRaw'], 'name': record['JobName'],
            'state': record['State'], 'exit_code': record['ExitCode'],
            'elapsed_seconds': int(record['ElapsedRaw']),
            'gpu_hours': record['allocated_gpu_hours_observed'],
            'run_directories': ';'.join(record['run_directories']),
            'cause': 'Not inferred from exit code; inspect preserved run events and Slurm logs.'})
    args.output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(args.output/'unsuccessful-allocations.csv', index=False)
    (args.output/'manifest.json').write_text(json.dumps({'status': 'interim_index',
        'source_sha256': file_hash(args.accounting), 'script_sha256': file_hash(Path(__file__)),
        'indexed_allocations': len(rows), 'gpu_hours': sum(r['gpu_hours'] for r in rows),
        'scope': 'Only allocations represented in input snapshot. Pending withdrawals without run directories, earlier campaigns and local transfer failures need separate records; refresh at campaign completion.'}, indent=2))


if __name__ == '__main__':
    main()
