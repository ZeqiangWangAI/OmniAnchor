import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile

import pytest

spec = importlib.util.spec_from_file_location('package_allowlist', Path(__file__).parents[1]/'scripts/package_allowlist.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_only_reviewed_bytes_are_archived(tmp_path):
    (tmp_path/'source.py').write_text('x = 1\n')
    (tmp_path/'restricted.txt').write_text('excluded')
    allowlist = tmp_path/'allowlist.json'
    allowlist.write_text(json.dumps({'files': {'source.py': hashlib.sha256((tmp_path/'source.py').read_bytes()).hexdigest()}}))
    output = tmp_path/'source.tgz'
    module.build(tmp_path, allowlist, output)
    with tarfile.open(output) as archive:
        assert archive.getnames() == ['source.py']
    with pytest.raises(FileExistsError):
        module.build(tmp_path, allowlist, output)
    (tmp_path/'source.py').write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        module.build(tmp_path, allowlist, tmp_path/'changed.tgz')
    assert not (tmp_path/'changed.tgz').exists()


def test_rejects_parent_escape(tmp_path):
    allowlist = tmp_path/'allowlist.json'
    allowlist.write_text(json.dumps({'files': {'../outside': 'unused'}}))
    with pytest.raises(ValueError, match='Unsafe'):
        module.build(tmp_path, allowlist, tmp_path/'unsafe.tgz')
