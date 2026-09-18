import numpy as np
import pytest

from scripts.analyze_viva import aligned_scores


def test_viva_alignment_and_missing_candidate(tmp_path):
    np.savez(tmp_path/"part-00000.npz", sample_id="recipient", anchor_ids=["b", "a"], scores=[.8, .2])
    rows, _ = aligned_scores(tmp_path, False, {"recipient": ["a", "b"]})
    np.testing.assert_array_equal(rows["recipient"], [.2, .8])
    with pytest.raises(ValueError, match="Candidate inventory"):
        aligned_scores(tmp_path, False, {"recipient": ["a", "c"]})
    with pytest.raises(ValueError, match="Incomplete recipient"):
        aligned_scores(tmp_path, False, {"recipient": ["a", "b"], "missing": ["a", "b"]})
