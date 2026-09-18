import hashlib

import pytest

from scripts.verify_multilabel_final import read_frozen_feature_info


def test_local_mirror_requires_original_metadata_hash(tmp_path):
    source = tmp_path/'features.json'
    source.write_text('{"native": {}}')
    original = '/unavailable/hpc/features'
    contract = {'probe_features': original, 'frozen_artifacts': {
        original+'/features.json': hashlib.sha256(source.read_bytes()).hexdigest()}}
    assert read_frozen_feature_info(contract, tmp_path) == {'native': {}}
    source.write_text('{"different": {}}')
    with pytest.raises(ValueError, match='metadata changed'):
        read_frozen_feature_info(contract, tmp_path)


def test_original_feature_path_still_supported(tmp_path):
    source = tmp_path/'features.json'
    source.write_text('{}')
    contract = {'probe_features': str(tmp_path), 'frozen_artifacts': {
        str(source): hashlib.sha256(source.read_bytes()).hexdigest()}}
    assert read_frozen_feature_info(contract) == {}
