import numpy as np
import pytest

from scripts.analyze_vatex_final import paired_interval, validate_family


def test_paired_zero_is_not_significant_and_group_units_preserved():
    result = paired_interval(np.zeros(6), ['a', 'a', 'a', 'b', 'c', 'c'])
    assert result['centered_bootstrap_p'] == 1
    assert result['lower'] == result['upper'] == 0
    assert result['independent_groups'] == 3
    positive = paired_interval(np.ones(6), ['a', 'a', 'a', 'b', 'c', 'c'])
    assert positive['centered_bootstrap_p'] == 1/1001
    assert positive['lower'] == positive['upper'] == 1


def test_family_rejects_duplicates_unknown_methods_and_reverse_primary():
    row = dict(language='en', method_a='native', method_b='baseline',
               direction='caption_to_video', metric='recall@1')
    assert len(validate_family([row], ['native', 'baseline'])) == 1
    with pytest.raises(ValueError, match='duplicated'):
        validate_family([row, row], ['native', 'baseline'])
    with pytest.raises(ValueError, match='unknown'):
        validate_family([row], ['native'])
    with pytest.raises(ValueError, match='caption-to-video'):
        validate_family([{**row, 'direction': 'video_to_caption'}], ['native', 'baseline'])
