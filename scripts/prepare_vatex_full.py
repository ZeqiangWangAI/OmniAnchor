"""Stage complete VATEX development/final scoring after the original32-input pilots."""
import argparse
from datetime import datetime, timezone
import os
import json
from pathlib import Path

from vlanchor.campaign import create_run, append_event
from vlanchor.io import load_samples, read_json, write_json
from vlanchor.provenance import file_hash


def stage(samples, path):
    """Keep media paths portable when the same frozen source is copied to scratch."""
    rows = []
    for sample in samples:
        parts = tuple(part.model_copy(update={'path': os.path.relpath(part.path, path.parent)})
                      if part.path else part for part in sample.parts)
        rows.append(sample.model_copy(update={'parts': parts}))
    write_json(path, rows)
    if [s.model_dump() for s in load_samples(path)] != [s.model_dump() for s in samples]:
        raise ValueError('Staging altered the scoring inputs.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['output', 'native-pilot', 'baseline-pilot', 'reference']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    for name in ['output','native_pilot','baseline_pilot','reference']:
        setattr(args,name,getattr(args,name).resolve())
    root = Path(__file__).resolve().parents[1]
    data = root/'data/prepared/vatex-scoring-20260910-01'
    config = root/'configs/studies/vatex_general128.json'
    plan = root/'research/contracts/vatex-analysis-plan-20260911-01.json'
    pilot = data/'smoke32.json'
    expected = [s.id for s in load_samples(pilot)]
    if len(expected) != 32 or any(s.metadata['split'] != 'dev' for s in load_samples(pilot)):
        raise ValueError('Require exact32development pilot inputs.')
    folders = [args.native_pilot/'measurement', *[args.baseline_pilot/m for m in ['qwen-embedding', 'qwen-reranker']]]
    costs = {}
    for folder in folders:
        if (folder.parent/'exit_code.txt').read_text().strip() != '0':
            raise ValueError('Pilot failed or incomplete.')
        manifest = read_json(folder/'manifest.json')
        # Native and official wrappers use different names for the same input fingerprint.
        if manifest.get('samples_sha256') != file_hash(pilot) or manifest['spec_sha256'] != file_hash(config):
            raise ValueError('Pilot population bytes changed.')
        events = [json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
        if events[-1]['status'] != 'completed' or [e['sample_id'] for e in events if e['status']=='sample_completed'] != expected:
            raise ValueError('Incomplete pilot sample coverage.')
        cost = read_json(folder/'cost.json')
        if cost['samples'] != 32 or cost['anchors'] != 128:
            raise ValueError('Pilot is not the full128-anchor path.')
        costs[str(folder.relative_to(root))] = cost
    if read_json(args.native_pilot/'measurement/native-verification.json')['status'] != 'passed':
        raise ValueError('Native numerical acceptance failed.')
    if (args.reference/'exit_code.txt').read_text().strip() != '0':
        raise ValueError('Original64train reference incomplete.')
    reference_manifest = read_json(args.reference/'measurement/manifest.json')
    if reference_manifest['samples_sha256'] != file_hash(data/'reference64.json') or reference_manifest['spec_sha256'] != file_hash(config):
        raise ValueError('Reference input or instrument changed.')
    sources = ['src/vlanchor/engine.py','src/vlanchor/backends/hf.py','src/vlanchor/backends/media.py',
        'src/vlanchor/backends/vision_reuse.py','src/vlanchor/official_baselines.py',
        'scripts/run_measurement_shard.py','scripts/run_baseline_shard.py','scripts/frozen_evaluation.py',
        'scripts/vision_reuse_admission.py','scripts/verify_native.py','configs/models-20260910.json']
    populations = {split: {role: load_samples(data/f'{split}-{role}.json') for role in ['videos','en','zh']} for split in ['dev','test']}
    groups = {split: {s.group_id for s in rows['videos']} for split,rows in populations.items()}
    if groups['dev'] & groups['test']:
        raise ValueError('YouTube source leakage across partitions.')
    create_run(args.output, dict(purpose='Complete VATEX scoring preparation; no retrieval results inspected',
        analysis_plan_sha256=file_hash(plan), source_sha256={p:file_hash(root/p) for p in sources},
        pilot_costs=costs, hardware='All full runs RTX5000Ada, matching original reference45764 and frame studies; conservative shard budgets use measured A5000pilot costs',
        budget='16videos or256captions per shard;3h per job;1GPU; array concurrency1; no implicit reduction',
        acceleration='uncached teacher forcing, admitted vision reuse44786; media preprocessing reuse off',
        admission='Original32data/model paths completed onA5000; native train reference64 and multimodal numerical gate onAda complete. Record per-run actual cost/hardware; no claim that pilot cost was measured onAda.'))
    try:
        for split, roles in populations.items():
            count = 93 if split=='dev' else 407
            if len(roles['videos']) != count or any(len(roles[language]) != 10*count for language in ['en','zh']):
                raise ValueError('Incomplete original frozen population.')
            files, contracts, ordered = [], {}, []
            for role, samples in roles.items():
                stage(samples, args.output/split/f'{role}.json')
                size = 16 if role=='videos' else 256
                for offset in range(0,len(samples),size):
                    selected = samples[offset:offset+size]
                    index = len(files)
                    path = args.output/split/'shards'/f'samples-{index:05d}.json'
                    stage(selected,path)
                    ordered.extend(s.id for s in selected)
                    files.append(dict(file=path.name,sha256=file_hash(path),sample_ids=[s.id for s in selected],role=role))
                    if split=='test':
                        admission = path.with_name(f'contract-{index:05d}.json')
                        write_json(admission,dict(status='frozen',frozen_utc=datetime.now(timezone.utc).isoformat(),
                            purpose='One shard of complete protected VATEX scoring',methods=['native','qwen-embedding','qwen-reranker'],
                            samples_sha256=file_hash(path),sample_ids=[s.id for s in selected],config_sha256=file_hash(config),
                            media_sha256={s.id:[dict(part_index=i,sha256=file_hash(p.path)) for i,p in enumerate(s.parts) if p.path] for s in selected},
                            verification_samples=str(pilot.relative_to(root)),verification_samples_sha256=file_hash(pilot),
                            scoring_source_sha256={p:file_hash(root/p) for p in sources},hardware='gpu_5000_ada',
                            budgets=dict(wall_hours=3,gpu_per_job=1),shared_prefill=False,media_preprocessing_reuse=False,
                            analysis_plan_sha256=file_hash(plan),model_adaptation='none'))
                        contracts[str(admission.relative_to(args.output/split))]=file_hash(admission)
            shard_manifest=args.output/split/'shards/manifest.json'
            write_json(shard_manifest,dict(files=files,train_samples=0,dev_samples=len(ordered) if split=='dev' else 0,test_samples=len(ordered) if split=='test' else 0))
            if split=='test':
                ref_files=sorted((args.reference/'measurement').glob('part-*.parquet*'))
                write_json(args.output/split/'feature-contract.json',dict(status='frozen',sample_ids=ordered,
                    shards_manifest_sha256=file_hash(shard_manifest),baseline_methods=['qwen-embedding','qwen-reranker'],
                    reference_score_sha256={str(p.relative_to(args.reference)):file_hash(p) for p in ref_files},
                    feature_assembly_sha256=file_hash(root/'scripts/build_development_features.py'),
                    scoring_contract_sha256=contracts,analysis_plan_sha256=file_hash(plan)))
        append_event(args.output,'completed')
    except BaseException as exc:
        append_event(args.output,'failed',error=repr(exc))
        raise


if __name__=='__main__':
    main()
