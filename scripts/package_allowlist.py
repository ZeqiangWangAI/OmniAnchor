"""Build a source archive from an explicit, hash-locked file allowlist."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def build(root: Path, allowlist: Path, output: Path) -> None:
    entries = json.loads(allowlist.read_text())['files']
    checked = []
    for name, expected in entries.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'Unsafe archive path: {name}')
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f'Path escapes source root: {name}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'File changed after allowlist review: {name}')
        checked.append((name, path))
    with output.open('xb') as stream:
        with tarfile.open(fileobj=stream, mode='w:gz') as archive:
            for name, path in sorted(checked):
                archive.add(path, arcname=name, recursive=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--allowlist', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.root, args.allowlist, args.output)
