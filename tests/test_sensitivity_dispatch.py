import json

import pytest

from scripts.run_sensitivity_analysis import match_runs


def test_match_instruments_by_frozen_spec_not_scheduler_order(tmp_path):
    rows = [dict(variant=name, spec_sha256=name, reference_runs=['reference-'+name]) for name in ['a','b']]
    runs = []
    for name in ['b','a']:
        run = tmp_path/name
        (run/'measurement').mkdir(parents=True)
        (run/'measurement/manifest.json').write_text(json.dumps(dict(spec_sha256=name)))
        runs.append(run)
    result = match_runs(rows, runs)
    assert result['a']['runs'] == [str(tmp_path/'a')]
    assert result['b']['reference_runs'] == ['reference-b']
    with pytest.raises(ValueError, match='Incomplete'):
        match_runs(rows, runs[:1])
    with pytest.raises(ValueError, match='repeated'):
        match_runs(rows, runs+[runs[0]])
