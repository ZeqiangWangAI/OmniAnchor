"""Admit unchanged E3 train references from completed per-instrument32-input pilots."""
import argparse
import json
import math
from pathlib import Path
import shutil

from omnianchor.campaign import append_event, create_run
from omnianchor.io import load_samples, read_json, write_json
from omnianchor.provenance import file_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['bank','pilot-runs','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--study',choices=['emobank','dwug'],required=True)
    args=parser.parse_args()
    bank=read_json(args.bank/'manifest.json')
    pilots=read_json(args.pilot_runs)
    definition=bank['studies'][args.study]
    if set(pilots)!=set(definition['variants']):
        raise ValueError('Every predefined instrument needs its own completed pilot.')
    for name,digest in bank['files_sha256'].items():
        if file_hash(args.bank/name)!=digest:
            raise ValueError('Sensitivity inventory changed.')
    folder=args.bank/args.study
    reference=load_samples(folder/'reference.json')
    if len(reference)!=64 or any(s.metadata['split']!='train' for s in reference):
        raise ValueError('Use the original64train references.')
    evidence={}
    for name,run in pilots.items():
        run=Path(run)
        measurement=run/'measurement'
        if (run/'exit_code.txt').read_text().strip()!='0':
            raise ValueError('Pilot incomplete or failed: '+str(run))
        manifest=read_json(measurement/'manifest.json')
        if (manifest['samples_sha256']!=file_hash(folder/'smoke32.json')
                or manifest['spec_sha256']!=file_hash(folder/(name+'.json')) or manifest['shared_prefill']):
            raise ValueError('Pilot used a different input/instrument/path.')
        if '5000 Ada' not in read_json(measurement/'hardware.json')['gpu']:
            raise ValueError('Reference budget requires the matchedAda pilot.')
        if read_json(measurement/'native-verification.json')['status']!='passed':
            raise ValueError('Native numerical checks failed.')
        events=[json.loads(line) for line in (measurement/'events.jsonl').read_text().splitlines()]
        ids=[e['sample_id'] for e in events if e['status']=='sample_completed']
        if events[-1]['status']!='completed' or ids!=definition['smoke_ids']:
            raise ValueError('Pilot coverage differs from all32frozen IDs.')
        cost=read_json(measurement/'cost.json')
        if cost['samples']!=32 or cost['anchors']!=definition['variants'][name]['anchors']:
            raise ValueError('Pilot cost is not for the complete instrument.')
        hours=max(1,math.ceil((cost['total_seconds']*2*1.5+120)/3600))
        if hours>4:
            raise ValueError('Reference estimate exceeds admitted Slurm walltime; prepare explicit shards.')
        evidence[name]=dict(run=str(run),cost=cost,wall_hours=hours,
            hashes={str(p):file_hash(p) for p in [measurement/'manifest.json',measurement/'cost.json',measurement/'native-verification.json',measurement/'hardware.json']})
    create_run(args.output,dict(purpose='Unchanged train-only reference scoring for each admitted E3 instrument',
        study=args.study,inventory_sha256=file_hash(args.bank/'manifest.json'),pilot_evidence=evidence,
        reference_ids=definition['reference_ids'],test_used=False,
        policy='Ada,BF16,batch1,uncached teacher forcing; no event/anchor/model changes; protected128scoring waits for completed and frozen reference evidence'))
    try:
        rows=[]
        for index,name in enumerate(definition['variants']):
            config=args.output/f'spec-{index:05d}.json'
            samples=args.output/f'samples-{index:05d}.json'
            shutil.copyfile(folder/(name+'.json'),config)
            shutil.copyfile(folder/'reference.json',samples)
            rows.append(dict(index=index,variant=name,samples_sha256=file_hash(samples),spec_sha256=file_hash(config),wall_hours=evidence[name]['wall_hours']))
        write_json(args.output/'shards.json',dict(rows=rows,study=args.study,inventory_sha256=file_hash(args.bank/'manifest.json')))
        append_event(args.output,'completed')
    except BaseException as exc:
        append_event(args.output,'failed',error=repr(exc))
        raise


if __name__=='__main__':
    main()
