"""Preserve the complete DWUG development array and official baseline after completion."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

root = Path('/mnt/fast/nobackup/scratch4weeks/zw00924/VLanchor-20260910')
array_rows = subprocess.check_output(['sacct', '-j', '44791', '-n', '-P', '-X',
    '--format=JobIDRaw,JobID,State,ExitCode'], text=True)
resolved = {}
for line in array_rows.splitlines():
    columns = line.split('|')
    if len(columns) >= 4 and columns[0].isdigit() and columns[1].startswith('44791_'):
        index = int(columns[1].split('_')[1])
        assert columns[2:4] == ['COMPLETED', '0:0'], columns
        assert index not in resolved
        resolved[index] = columns[0]
assert set(resolved) == set(range(19)), 'Require all 19 development shards; no partial archive.'
jobs = [resolved[index] for index in range(19)] + ['45022']
archive = root/'dwug-development-raw-evidence-20260911-01.tgz'
record_path = archive.with_suffix('.json')
assert not archive.exists() and not record_path.exists()
states = subprocess.check_output(['sacct', '-j', ','.join(jobs), '-n', '-P', '-X',
                                  '--format=JobIDRaw,State,ExitCode'], text=True)
observed = {}
for line in states.splitlines():
    columns = line.split('|')
    if columns[0] in jobs:
        assert columns[1:3] == ['COMPLETED', '0:0'], columns
        observed[columns[0]] = columns[1:3]
assert set(observed) == set(jobs)
files, folders = {}, []
for job in jobs:
    matches = [p for p in (root/'runs').glob('*-'+job) if p.is_dir()]
    assert len(matches) == 1, matches
    folder = matches[0]
    assert (folder/'exit_code.txt').read_text().strip() == '0'
    folders.append(str(folder.relative_to(root)))
    for path in folder.rglob('*'):
        if any(part in {'source', 'media', 'venv', 'cache'} for part in path.relative_to(folder).parts):
            continue
        if path.is_file():
            files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
with tarfile.open(archive, 'w:gz') as output:
    for name in sorted(files):
        output.add(root/name, arcname=name)
record = {'runs': folders, 'files_sha256': files, 'jobs': observed,
          'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
          'archive_bytes': archive.stat().st_size,
          'scope': 'Private raw evidence, source/media duplicated trees excluded; no redistribution authorization.'}
record_path.write_text(json.dumps(record, indent=2))
print(record_path, len(files), record['archive_bytes'], record['archive_sha256'], flush=True)
