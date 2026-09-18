import json

import pytest

from scripts.prepare_viva_interaction_final import pilot_evidence


def test_interaction_admission_requires_complete_matched_pilot(tmp_path):
    folder = tmp_path/'native'
    folder.mkdir()
    ids = [f'i{i}' for i in range(32)]
    pilot = dict(conditions=['image_only', 'empty'], samples='smoke.json', samples_sha256='inputs',
        sources_sha256={'scripts/run_viva_shard.py': 'driver', 'scripts/viva_conditions.py': 'conditions'})
    manifest = dict(method='native', sample_ids=ids, conditions=pilot['conditions'],
        source_sha256='driver', condition_source_sha256='conditions', input_sha256={'smoke.json': 'inputs'})
    (folder/'manifest.json').write_text(json.dumps(manifest))
    (folder/'cost.json').write_text(json.dumps(dict(samples=32, gpu='RTX 5000 Ada', measurement_seconds=372)))
    (folder/'native-verification.json').write_text('{"status":"passed"}')
    (tmp_path/'exit_code.txt').write_text('0\n')
    events = [dict(status='sample_completed', sample_id=i) for i in ids]+[dict(status='completed')]
    (folder/'events.jsonl').write_text('\n'.join(json.dumps(e) for e in events))
    result = pilot_evidence(folder, 'native', ids, pilot)
    assert result['wall_hours'] == 2
    events.pop(0)
    (folder/'events.jsonl').write_text('\n'.join(json.dumps(e) for e in events))
    with pytest.raises(ValueError, match='all32'):
        pilot_evidence(folder, 'native', ids, pilot)
