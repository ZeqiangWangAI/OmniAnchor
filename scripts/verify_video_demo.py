"""Validate bounded video capability outputs and costs without retrieval claims."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['evidence', 'samples', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    samples = json.loads(args.samples.read_text())
    expected = {r['id'] for r in samples}
    assert len(expected) == 32
    counts = {kind: sum(any(p['type'] == kind for p in r['parts']) for r in samples) for kind in ['video', 'text']}
    assert counts == {'video': 16, 'text': 16}
    rows, hashes = [], {}
    for frames, native, baseline in [(8, '47130', '47146'), (16, '46727', '46728')]:
        for method in ['native', 'qwen-embedding', 'qwen-reranker']:
            run = args.evidence/'runs'/('development-'+native if method == 'native' else 'baseline-development-'+baseline)
            assert (run/'exit_code.txt').read_text().strip() == '0'
            folder = run/('measurement' if method == 'native' else method)
            cost = json.loads((folder/'cost.json').read_text())
            assert cost['samples'] == 32 and cost['anchors'] == 128
            if method == 'native':
                paths = sorted(folder.glob('part-*.parquet'))
                data = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
                assert set(data.sample_id) == expected and len(data) == 32*128*3
                assert not data.duplicated(['sample_id', 'anchor_id', 'bridge_id']).any()
                assert np.isfinite(data.raw_logp).all()
            else:
                paths = sorted(folder.glob('part-*.npz')); seen = set()
                for path in paths:
                    data = np.load(path); identifier = str(data['sample_id'].item())
                    assert identifier not in seen; seen.add(identifier)
                    assert data['scores'].shape == (128,) and np.isfinite(data['scores']).all()
                    assert len(set(data['anchor_ids'].tolist())) == 128
                assert seen == expected
            for path in paths+[folder/'cost.json', folder/'manifest.json']:
                hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append(dict(frames=frames, method=method, mixed_inputs=32, videos=16, texts=16,
                measurement_seconds=cost['measurement_seconds'], peak_allocated_bytes=cost['peak_allocated_bytes']))
    failures = []
    for run, method in [('development-47125', 'measurement'), ('baseline-development-47141', 'qwen-embedding')]:
        path = args.evidence/'runs'/run
        assert (path/'exit_code.txt').read_text().strip() == '1'
        event = json.loads((path/method/'events.jsonl').read_text().splitlines()[-1])
        assert event['status'] == 'failed' and 'temporal patch size' in event['error']
        failures.append(dict(run=run, error=event['error']))
    pd.DataFrame(rows).to_csv(args.output/'costs.csv', index=False)
    (args.output/'manifest.json').write_text(json.dumps({'status': 'passed', 'scope': 'Small-sample execution/finite score coverage only; no retrieval or scientific validity claim',
        'failures_preserved': failures, 'source_sha256': hashes}, indent=2))


if __name__ == '__main__':
    main()
