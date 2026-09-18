"""Summarize unique Slurm allocations, separating observed and inferred GPU hardware."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

from vlanchor.campaign import create_run, append_event
from vlanchor.io import read_json, write_json
from vlanchor.provenance import file_hash


def aggregate(rows):
    if len({r['JobIDRaw'] for r in rows})!=len(rows):
        raise ValueError('Duplicate job allocations would double-count costs.')
    totals=defaultdict(lambda: dict(jobs=0,gpu_hours=0.,cpu_hours=0.))
    for row in rows:
        if not row['JobIDRaw'].isdigit():
            raise ValueError('Job-step rows must not be counted as allocations.')
        target=totals[row['State']]
        target['jobs']+=1
        target['gpu_hours']+=row['allocated_gpu_hours_observed']
        target['cpu_hours']+=row['allocated_cpu_hours_observed']
    return dict(totals)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    rows=read_json(args.snapshot/'accounting.json')
    totals=aggregate(rows)
    telemetry=read_json(args.snapshot/'telemetry.json')
    devices=defaultdict(set)
    for item in telemetry:
        value=item['value']
        if isinstance(value,dict) and isinstance(value.get('gpu'),str):
            devices[item['job_id']].add(value['gpu'])
    hardware=defaultdict(lambda:dict(jobs=0,gpu_hours=0.))
    for row in rows:
        observed=devices[row['JobIDRaw']]
        if row['allocated_gpus']==0:
            label='CPU allocation; no GPU requested'
        elif len(observed)==1:
            label='Observed: '+next(iter(observed))
        elif len(observed)>1:
            raise ValueError('Conflicting hardware evidence in one allocation.')
        else:
            # The raw node inventory is retained; inference is never labelled an observation.
            node=row['NodeList']
            inventory=(args.snapshot/'current-node-inventory.psv').read_text().splitlines()
            features=' '.join(line for line in inventory if line.split('|')[0]==node)
            kind='RTX 5000 Ada' if 'gpu_5000_ada' in features else 'RTX A5000' if 'gpu_a5000' in features else 'unresolved'
            label='Node-inventory inference: '+kind
        hardware[label]['jobs']+=1
        hardware[label]['gpu_hours']+=row['allocated_gpu_hours_observed']
    paths=[args.snapshot/name for name in ['manifest.json','accounting.json','telemetry.json','slurm-accounting.psv','current-node-inventory.psv']]
    create_run(args.output,dict(purpose='Interim campaign cost audit; no score or label reads',source_sha256={str(p):file_hash(p) for p in paths},script_sha256=file_hash(Path(__file__))))
    write_json(args.output/'summary.json',dict(job_records=len(rows),state_counts=dict(Counter(r['State'] for r in rows)),
        observed_allocations_by_state=totals,hardware=hardware,
        total_observed_allocated_gpu_hours=sum(r['allocated_gpu_hours_observed'] for r in rows),
        scope='Unique numbered campaign jobs only, including failed and right-censored running allocations. Excludes pending jobs without run directories and historical runs outside this scratch campaign. Not final cost or remaining-budget prediction.',
        timing='Allocated time includes model load, verification and idle overhead; telemetry measurement times are separate, not additive to allocation hours.'))
    append_event(args.output,'completed')


if __name__=='__main__':
    main()
