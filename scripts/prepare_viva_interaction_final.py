"""Admit unchanged VIVA image-only/empty controls after complete matched32-input pilots."""
import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from omnianchor.io import load_samples, read_json, write_json
from omnianchor.provenance import file_hash


def pilot_evidence(folder, method, expected_ids, pilot):
    manifest = read_json(folder/'manifest.json')
    cost = read_json(folder/'cost.json')
    events = [json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
    if ((folder.parent/'exit_code.txt').read_text().strip() != '0' or not events
            or events[-1]['status'] != 'completed' or manifest['method'] != method
            or manifest['sample_ids'] != expected_ids or cost['samples'] != 32
            or [e['sample_id'] for e in events if e['status'] == 'sample_completed'] != expected_ids):
        raise ValueError('Require all32 pilot inputs completed exactly once.')
    if manifest['conditions'] != pilot['conditions'] or '5000 Ada' not in cost['gpu']:
        raise ValueError('Pilot conditions or hardware differ.')
    if (manifest['source_sha256'] != pilot['sources_sha256']['scripts/run_viva_shard.py']
            or manifest['condition_source_sha256'] != pilot['sources_sha256']['scripts/viva_conditions.py']
            or manifest['input_sha256'][pilot['samples']] != pilot['samples_sha256']):
        raise ValueError('Pilot input or scoring identity differs.')
    if method == 'native' and read_json(folder/'native-verification.json')['status'] != 'passed':
        raise ValueError('Native numerical check failed.')
    return dict(run=str(folder), cost=cost,
        input_sha256={str(p): file_hash(p) for p in [folder/'manifest.json', folder/'cost.json', folder/'events.jsonl']},
        wall_hours=max(1, math.ceil((cost['measurement_seconds']*216/32*1.5+120)/3600)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['pilot-contract', 'pilot-runs', 'base-contract', 'samples', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Do not overwrite frozen admissions.')
    pilot, base = read_json(args.pilot_contract), read_json(args.base_contract)
    runs = read_json(args.pilot_runs)
    ids = [s.id for s in load_samples(pilot['samples'])]
    if len(ids) != 32 or set(runs) != set(base['methods']) or file_hash(args.samples) != base['samples_sha256']:
        raise ValueError('Require unchanged pilot and original final populations.')
    for path, digest in pilot['sources_sha256'].items():
        if file_hash(path) != digest:
            raise ValueError('Pilot scoring or interaction definition changed.')
    evidence = {method: pilot_evidence(Path(runs[method]), method, ids, pilot) for method in base['methods']}
    hours = max(e['wall_hours'] for e in evidence.values())
    if hours > 3:
        raise ValueError('Measured budget requires explicit smaller shards.')
    contract = dict(base)
    contract.update(frozen_utc=datetime.now(timezone.utc).isoformat(),
        purpose='Prespecified VIVA output-score interaction; additional image-only and empty conditions',
        conditions=pilot['conditions'], pilot_evidence=evidence,
        base_contract_sha256=file_hash(args.base_contract),
        interaction_analysis_sha256=file_hash('scripts/analyze_viva_interaction.py'),
        primary='Within-method signed delta and recipient RMS delta; no cross-method magnitude comparison',
        inference='1000recipient-image-group descriptive95%CI; no superiority or internalfusion test',
        method_adaptation='None; same216recipients, anchors, models, pixels and resource budgets. Development AP outcomes known before this admission; predefined interaction definition unchanged.',
        budget=dict(tasks=3, wall_hours_per_task=hours, max_concurrent=1))
    for path in set(base['scoring_source_sha256']) | {'scripts/viva_conditions.py'}:
        contract['scoring_source_sha256'][path] = file_hash(path)
    write_json(args.output, contract)


if __name__ == '__main__':
    main()
