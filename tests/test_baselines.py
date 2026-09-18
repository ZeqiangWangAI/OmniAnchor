import sys
from types import SimpleNamespace

import numpy as np
import pytest

from omnianchor.baselines import HFCrossEncoder, HFMeanEmbedding, HFSingleTokenMLM, cosine_scores
from omnianchor.errors import ResourceUnavailable


def test_cosine_baseline_axis_and_zero_vector_failure():
    result = cosine_scores([[1, 0], [0, 1]], [[1, 0], [-1, 0]])
    np.testing.assert_allclose(result, [[1, -1], [0, 0]])
    with pytest.raises(ValueError, match="zero"):
        cosine_scores([[0, 0]], [[1, 0]])


def test_baselines_instantiate_without_importing_or_loading_models():
    models = [HFMeanEmbedding("local/model", "commit"), HFCrossEncoder("local/model", "commit"),
              HFSingleTokenMLM("local/model", "commit")]
    assert all(model._model is None and model.local_files_only for model in models)
    assert models[2].capabilities["anchor_tokens"] == 1
    assert not models[0].capabilities["supports_images"]
    with pytest.raises(ValueError):
        HFMeanEmbedding("model", "")


def test_requested_cuda_never_falls_back_to_cpu(monkeypatch):
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace())
    baseline = HFMeanEmbedding("local/model", "commit", device="cuda")
    with pytest.raises(ResourceUnavailable, match="no CPU fallback"):
        baseline._load("AutoModel")
