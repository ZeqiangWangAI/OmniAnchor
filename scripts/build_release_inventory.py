"""Inventory tracked release candidates; file presence is not redistribution approval."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=False)
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
    rows = []
    for name in sorted(n for n in tracked if n):
        path = root/name
        if path.is_symlink():
            raise ValueError('Symlink needs explicit release review: '+name)
        prefix = Path(name).parts[0]
        if prefix in {'src', 'scripts', 'tests'}:
            category, review = 'source_or_tests', 'review third-party notices and embedded fixtures'
        elif prefix == 'configs':
            category, review = 'measurement_configuration', 'review embedded source material; preserve instrument semantics'
        elif prefix == 'locks':
            category, review = 'environment_provenance', 'distinguish observed inventory from verified clean installation'
        elif prefix == 'paper':
            category, review = 'manuscript_or_derived_result', 'review source rights and final claim/evidence correspondence'
        elif prefix == 'research':
            category, review = 'private_evidence_record', 'exclude by default; only explicit reviewed metadata exports'
        else:
            category, review = 'documentation', 'review embedded data and operational paths'
        rows.append(dict(path=name, category=category, bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), review=review))
    pd.DataFrame(rows).to_csv(args.output/'tracked-file-inventory.csv', index=False)
    (args.output/'manifest.json').write_text(json.dumps({'status':'inventory_not_approved_release',
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'tracked_files':len(rows), 'category_counts':pd.Series([r['category'] for r in rows]).value_counts().to_dict(),
        'excluded_untracked_roots':['data','runs','reports','.venv'],
        'remaining':'Review an explicit per-file public allowlist; separately inventory eligible raw score/calibration/network artifacts; verify complete reproduction before final packaging.',
        'scope':'Private assembly inventory only; does not authorize redistribution or certify final task completion'},indent=2))


if __name__ == '__main__':
    main()
