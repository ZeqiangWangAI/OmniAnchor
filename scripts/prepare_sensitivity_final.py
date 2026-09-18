"""Freeze bounded E3 protected scoring only after every train reference is complete."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import math

from vlanchor.campaign import create_run, append_event
from vlanchor.io import load_samples, read_json, write_json
from vlanchor.provenance import file_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['bank','reference-runs','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--study',choices=['emobank','dwug'],required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    bank=read_json(args.bank/'manifest.json')
    references=read_json(args.reference_runs)
    definition=bank['studies'][args.study]
    if set(references)!=set(definition['variants']):
        raise ValueError('Every frozen instrument requires its own train reference.')
    for name,digest in bank['files_sha256'].items():
        if file_hash(args.bank/name)!=digest:
            raise ValueError('Frozen inventory changed.')
    folder=args.bank/args.study
    samples=load_samples(folder/'test128.json')
    if len(samples)!=128 or [s.id for s in samples]!=definition['test_ids'] or any(s.metadata['split']!='test' for s in samples):
        raise ValueError('Require exactly the original128protected materials.')
    if set(definition['test_ids']) & set(definition['reference_ids']):
        raise ValueError('Reference and protected IDs overlap.')
    # Resolve the existing reader in both CLI and package/test execution modes.
    import sys
    sys.path.insert(0,str(root/'scripts'))
    from analyze_sensitivity import read_instrument
    reference_hashes={}
    wall_hours={}
    for name,runs in references.items():
        table,hashes=read_instrument(runs,definition['reference_ids'],file_hash(folder/(name+'.json')))
        if any(s['metadata']['split']!='train' for s in table.manifest['samples']):
            raise ValueError('Reference contains protected material.')
        reference_hashes[name]=hashes
        seconds=sum(read_json(Path(run)/'measurement/cost.json')['total_seconds'] for run in runs)
        wall_hours[name]=max(1,math.ceil((3*seconds+120)/3600))
        if wall_hours[name]>4:
            raise ValueError('Measured cost needs explicit smaller shards; no silent budget reduction.')
    sources=['src/vlanchor/engine.py','src/vlanchor/backends/hf.py','src/vlanchor/backends/media.py',
        'src/vlanchor/backends/vision_reuse.py','scripts/run_measurement_shard.py','scripts/frozen_evaluation.py',
        'scripts/verify_native.py','configs/models-20260910.json']
    analysis_sources=['scripts/analyze_sensitivity.py','scripts/sensitivity_statistics.py',
        'src/vlanchor/calibration.py','src/vlanchor/evaluation.py','src/vlanchor/analysis.py']
    create_run(args.output,dict(purpose='Freeze full bounded E3 sensitivity evaluation after train reference completion',
        study=args.study,inventory_sha256=file_hash(args.bank/'manifest.json'),reference_files_sha256=reference_hashes,
        timing='All references completed before protected scoring; no changes to previously fixed contrasts or selected128IDs'))
    try:
        contracts,rows={},[]
        for index,name in enumerate(definition['variants']):
            config=args.output/f'spec-{index:05d}.json'
            path=args.output/f'samples-{index:05d}.json'
            admission=args.output/f'contract-{index:05d}.json'
            shutil.copyfile(folder/(name+'.json'),config)
            shutil.copyfile(folder/'test128.json',path)
            pilot=folder/'smoke32.json'
            write_json(admission,dict(status='frozen',frozen_utc=datetime.now(timezone.utc).isoformat(),
                purpose='One complete128material E3 instrument evaluation',methods=['native'],sample_ids=definition['test_ids'],
                samples_sha256=file_hash(path),config_sha256=file_hash(config),media_sha256={s.id:[] for s in samples},
                verification_samples=str(pilot.relative_to(root)) if pilot.is_absolute() else str(pilot),
                verification_samples_sha256=file_hash(pilot),scoring_source_sha256={p:file_hash(root/p) for p in sources},
                reference_files_sha256=reference_hashes[name],hardware='gpu_5000_ada',shared_prefill=False,
                budgets=dict(wall_hours=wall_hours[name],gpu_per_job=1),method_adaptation='none; full fixed sensitivity instrument'))
            contracts[name]=[file_hash(admission)]
            rows.append(dict(index=index,variant=name,wall_hours=wall_hours[name],reference_runs=references[name],samples_sha256=file_hash(path),spec_sha256=file_hash(config),contract_sha256=file_hash(admission)))
        write_json(args.output/'shards.json',dict(rows=rows,study=args.study))
        write_json(args.output/'analysis-contract.json',dict(status='frozen',study=args.study,
            inventory_sha256=file_hash(args.bank/'manifest.json'),reference_files_sha256=reference_hashes,
            scoring_contract_sha256=contracts,analysis_source_sha256={p:file_hash(root/p) for p in analysis_sources},
            inference='1000sourcegroup descriptive intervals, fixedk2seed42 template clusters; no testmethodselection or superiority family'))
        append_event(args.output,'completed')
    except BaseException as exc:
        append_event(args.output,'failed',error=repr(exc))
        raise


if __name__=='__main__':
    main()
