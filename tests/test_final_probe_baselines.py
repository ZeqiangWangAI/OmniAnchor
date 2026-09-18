import pytest

from scripts.evaluate_text_final import baseline_methods


def test_historical_text_contract_keeps_all_three_baselines():
    assert baseline_methods({"methods": ["native", "e5", "qwen-embedding", "qwen-reranker"]}) == [
        "e5", "qwen-embedding", "qwen-reranker"]


def test_image_admission_requires_both_official_baselines():
    contract = {"methods": ["native", "qwen-embedding", "qwen-reranker"],
        "baseline_methods": ["qwen-embedding", "qwen-reranker"]}
    assert baseline_methods(contract) == contract["baseline_methods"]
    with pytest.raises(ValueError, match="complete declared"):
        baseline_methods({**contract, "baseline_methods": ["qwen-embedding"]})
    with pytest.raises(ValueError, match="disagree"):
        baseline_methods({**contract, "methods": ["native", "qwen-embedding"]})
