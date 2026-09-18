"""Check claim-source presence and frozen hashes; this is not a semantic claim audit."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--claims', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    document = json.loads(args.claims.read_text())
    rows = document['claims']
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate claim identifiers.')
    checks = []
    audits = document.get('independent_checks', [])
    entries = rows + ([{'id': 'independent_checks', 'evidence': audits}] if audits else [])
    for row in entries:
        evidence = row['evidence']
        sources = [{'path': evidence, 'sha256': row.get('sha256')}] if isinstance(evidence, str) else evidence
        if not sources:
            raise ValueError('Missing evidence.')
        for source in sources:
            path = Path(source['path'])
            if not path.is_file() or not source.get('sha256'):
                raise ValueError('Missing file or hash: '+str(path))
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != source['sha256']:
                raise ValueError('Evidence hash changed: '+str(path))
            checks.append({'claim': row['id'], 'path': str(path), 'sha256': actual})
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'verification.json').write_text(json.dumps({'status': 'passed', 'claims': len(rows),
        'file_references': len(checks), 'independent_check_records': len(audits), 'checks': checks,
        'scope': 'Artifact presence and recorded hash equality only; does not certify claim wording, statistics or completion of E1-E6'}, indent=2))


if __name__ == '__main__':
    main()
