import numpy as np
import pytest

from scripts.analyze_vatex import align_scores, retrieval_rows, two_stage_priority


def test_video_retrieval_success_is_not_fractional_recall():
    relevance = np.array([[1, 1, 0], [0, 0, 1]])
    scores = np.array([[3., 2., 0.], [2., 1., 3.]])
    rows = retrieval_rows(relevance, scores)
    np.testing.assert_equal(rows["success@1"], [1, 1])
    np.testing.assert_equal(rows["recall@1"], [.5, 1])
    np.testing.assert_equal(rows["recall@5"], [1, 1])
    with pytest.raises(ValueError, match="Missing paired"):
        retrieval_rows([[0, 0]], [[1., 2.]])


def test_retrieval_alignment_preserves_ids_and_rejects_missing(tmp_path):
    path = tmp_path/"scores.npz"
    np.savez(path, scores=[[4, 3], [2, 1]], query_ids=["b", "a"], candidate_ids=["y", "x"])
    np.testing.assert_equal(align_scores(path, ["a", "b"], ["x", "y"]), [[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="full frozen population"):
        align_scores(path, ["a"], ["x", "y"])


def test_two_stage_rank_keeps_unscored_probabilities_undefined():
    embedding = np.array([[.8, .7, .5, .2]])
    selected = np.array([[True, True, False, False]])
    probabilities = np.array([[.1, .9, np.nan, np.nan]])
    priority = two_stage_priority(embedding, probabilities, selected)
    assert np.argsort(-priority)[0].tolist() == [1, 0, 2, 3]
    assert np.isnan(probabilities[0, 2:]).all()
    with pytest.raises(ValueError, match="undefined"):
        two_stage_priority(embedding, np.nan_to_num(probabilities), selected)
