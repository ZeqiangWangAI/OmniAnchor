import pytest

from scripts.prepare_empty_predictions import empty_paths


def test_empty_layout_requires_both_complete_separate_runs(tmp_path):
    native,baseline=tmp_path/'native',tmp_path/'baseline'
    for folder in [native,baseline]:
        folder.mkdir()
        (folder/'exit_code.txt').write_text('0\n')
    assert empty_paths(None,native,baseline)==(native/'measurement',baseline)
    with pytest.raises(ValueError,match='Both separate'):
        empty_paths(None,native,None)
    (baseline/'exit_code.txt').write_text('1\n')
    with pytest.raises(ValueError,match='incomplete or failed'):
        empty_paths(None,native,baseline)
    with pytest.raises(ValueError,match='not both'):
        empty_paths(tmp_path/'packed',native,baseline)


def test_legacy_empty_layout_is_preserved(tmp_path):
    (tmp_path/'exit_code.txt').write_text('0\n')
    assert empty_paths(tmp_path/'en',None,None)==(tmp_path/'en/native',tmp_path/'en')
