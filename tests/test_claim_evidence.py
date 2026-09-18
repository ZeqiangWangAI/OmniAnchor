import hashlib
import json
import sys

import pytest

from scripts.verify_claim_evidence import main


@pytest.mark.parametrize('multiple', [False, True])
def test_claim_evidence_accepts_both_existing_formats(tmp_path, monkeypatch, multiple):
    source = tmp_path/'source.csv'
    source.write_text('value\n1\n')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    claim = {'id': 'one', 'evidence': [{'path': str(source), 'sha256': digest}] if multiple else str(source), 'sha256': digest}
    path = tmp_path/'claims.json'
    path.write_text(json.dumps({'claims': [claim]}))
    output = tmp_path/'checked'
    monkeypatch.setattr(sys, 'argv', ['verify', '--claims', str(path), '--output', str(output)])
    main()
    assert json.loads((output/'verification.json').read_text())['file_references'] == 1


def test_claim_evidence_rejects_changed_source(tmp_path, monkeypatch):
    source = tmp_path/'source.csv'
    source.write_text('changed')
    path = tmp_path/'claims.json'
    path.write_text(json.dumps({'claims': [{'id': 'one', 'evidence': str(source), 'sha256': hashlib.sha256(b'original').hexdigest()}]}))
    output = tmp_path/'checked'
    monkeypatch.setattr(sys, 'argv', ['verify', '--claims', str(path), '--output', str(output)])
    with pytest.raises(ValueError, match='hash changed'):
        main()
    assert not output.exists()


@pytest.mark.parametrize('changed', [False, True])
def test_independent_check_hashes_are_verified(tmp_path, monkeypatch, changed):
    source = tmp_path/'audit.json'
    source.write_text('original')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    path = tmp_path/'claims.json'
    path.write_text(json.dumps({'claims': [], 'independent_checks': [
        {'path': str(source), 'sha256': digest}]}))
    output = tmp_path/'checked'
    monkeypatch.setattr(sys, 'argv', ['verify', '--claims', str(path), '--output', str(output)])
    if changed:
        source.write_text('changed')
        with pytest.raises(ValueError, match='hash changed'):
            main()
        assert not output.exists()
    else:
        main()
        assert json.loads((output/'verification.json').read_text())['independent_check_records'] == 1
