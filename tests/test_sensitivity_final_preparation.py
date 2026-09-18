import json
import subprocess
import sys


def test_incomplete_reference_inventory_stops_before_protected_inputs(tmp_path):
    bank=tmp_path/'bank'
    bank.mkdir()
    (bank/'manifest.json').write_text(json.dumps({'studies':{'emobank':{'variants':{'base':{},'terminated':{}}}}}))
    refs=tmp_path/'references.json'
    refs.write_text(json.dumps({'base':['missing-run']}))
    output=tmp_path/'final'
    result=subprocess.run([sys.executable,'scripts/prepare_sensitivity_final.py','--bank',str(bank),
        '--reference-runs',str(refs),'--study','emobank','--output',str(output)],capture_output=True,text=True)
    assert result.returncode!=0
    assert 'Every frozen instrument requires its own train reference' in result.stderr
    assert not output.exists()
