"""Submit the admitted OASIS12 then full VATEX campaign from an immutable Surrey release."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--record',type=Path,required=True)
    args=parser.parse_args()
    if args.record.exists():
        raise FileExistsError('Submission record already exists; inspect job IDs before retrying.')
    root=Path('/mnt/fast/nobackup/scratch4weeks/zw00924/VLanchor-20260910')
    release=args.release.resolve()
    os.chdir(release)
    sys.path[:0]=[str(release/'src'),str(release/'scripts')]
    from frozen_evaluation import validate_frozen_evaluation
    from vision_reuse_admission import admit_vision_reuse
    from vlanchor.io import load_samples,read_json
    from vlanchor.provenance import file_hash
    gate=root/'runs/vision-reuse-gate-44786/gate/summary.json'
    admit_vision_reuse(gate)
    config=Path('configs/studies/vatex_general128.json')
    vatex=Path('data/prepared/vatex-full-20260911-01')
    admission=read_json(Path('research/contracts/vatex-full-scoring-20260911-01.json'))
    for name,digest in admission['inputs_sha256'].items():
        if file_hash(name)!=digest:
            raise ValueError('Full VATEX staging hash mismatch: '+name)
    for path in sorted((vatex/'test/shards').glob('samples-*.json')):
        contract=path.with_name(path.name.replace('samples-','contract-'))
        for method in ['native','qwen-embedding','qwen-reranker']:
            validate_frozen_evaluation(contract,path,config,method)
    oasis=Path('data/prepared/oasis-affect-20260910-01')
    for path in [oasis/'shards/samples-00000.json',oasis/'reference/samples.json']:
        samples=load_samples(path)
        if any(s.metadata['split'] not in {'train','dev'} or not Path(s.parts[0].path).is_file() for s in samples):
            raise ValueError('OASIS source role/media preflight failed.')
    feature_contract=read_json(vatex/'test/feature-contract.json')
    for name,digest in feature_contract['reference_score_sha256'].items():
        if file_hash(root/'runs/development-45764'/name)!=digest:
            raise ValueError('VATEX original reference hash mismatch.')
    env={k:v for k,v in os.environ.items() if not k.startswith('VL_')}
    env.update(VL_SOURCE_DIR=str(release),VL_ENV_PYTHON=str(root/'runs/smoke-44672/venv/bin/python'),
        VL_VISION_REUSE_GATE=str(gate),VL_METHODS='qwen-embedding qwen-reranker')
    record=dict(release=str(release),created_utc=datetime.now(timezone.utc).isoformat(),preflight='passed',jobs={})
    args.record.write_text(json.dumps(record,indent=2)+'\n')

    def submit(name,script,options,variables):
        command=['sbatch','--parsable',f'--output={root}/logs/{name}-%A-%a.out',f'--error={root}/logs/{name}-%A-%a.err',*options,script]
        job=subprocess.check_output(command,env={**env,**variables},text=True).strip().split(';')[0]
        record['jobs'][name]=dict(job_id=job,command=command,variables=variables)
        args.record.write_text(json.dumps(record,indent=2)+'\n')
        print(name,job,flush=True)
        return job

    native=submit('oasis12-native-full','scripts/hpc/development.sbatch',['--constraint=gpu_5000_ada','--time=04:00:00'],
        dict(VL_CONFIG='configs/smoke/qwen35-en.json',VL_SAMPLES=str(oasis/'shards/samples-00000.json')))
    baselines=submit('oasis12-baselines-full','scripts/hpc/baseline_development.sbatch',['--constraint=gpu_5000_ada','--time=03:00:00'],
        dict(VL_CONFIG='configs/smoke/qwen35-en.json',VL_SAMPLES=str(oasis/'shards/samples-00000.json')))
    reference=submit('oasis12-reference64','scripts/hpc/development.sbatch',['--constraint=gpu_5000_ada','--time=01:00:00'],
        dict(VL_CONFIG='configs/smoke/qwen35-en.json',VL_SAMPLES=str(oasis/'reference/samples.json')))
    request=root/'oasis12-features-request-20260911-01.json'
    if request.exists():
        raise FileExistsError('Preserve previous request.')
    request.write_text(json.dumps(dict(script='build_development_features.py',arguments=['--shards',str(oasis/'shards'),
        '--native-runs',str(root/f'runs/development-{native}'),'--baseline-runs',str(root/f'runs/baseline-development-{baselines}'),
        '--reference-run',str(root/f'runs/development-{reference}'),'--methods','qwen-embedding','qwen-reranker']),indent=2)+'\n')
    features=submit('oasis12-features','scripts/hpc/analysis.sbatch',[f'--dependency=afterok:{native}:{baselines}:{reference}'],dict(VL_ANALYSIS_REQUEST=str(request)))
    request=root/'oasis12-probes-request-20260911-01.json'
    if request.exists():
        raise FileExistsError('Preserve previous request.')
    request.write_text(json.dumps(dict(script='fit_development_probes.py',arguments=['--features',str(root/f'runs/analysis-{features}/analysis'),
        '--train',str(oasis/'probe_train'),'--dev',str(oasis/'dev'),'--task','regression']),indent=2)+'\n')
    submit('oasis12-probes','scripts/hpc/analysis.sbatch',[f'--dependency=afterok:{features}'],dict(VL_ANALYSIS_REQUEST=str(request)))
    for split,count in [('dev',14),('test',58)]:
        variables=dict(VL_CONFIG=str(config),VL_SHARD_DIR=str(vatex/split/'shards'))
        if split=='test':
            variables['VL_FROZEN_EVALUATION_DIR']=str(vatex/split/'shards')
        for method,script in [('native','development.sbatch'),('baselines','baseline_development.sbatch')]:
            submit(f'vatex-{split}-{method}-full','scripts/hpc/'+script,
                ['--constraint=gpu_5000_ada','--time=03:00:00',f'--array=0-{count-1}%1'],variables)
    record['submission_completed']=True
    args.record.write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':
    main()
