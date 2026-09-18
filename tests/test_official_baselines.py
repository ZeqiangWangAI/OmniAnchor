import numpy as np
import pytest

from vlanchor.errors import BudgetExceeded
from vlanchor.official_baselines import E5Embedding, require_cuda


def test_requires_immutable_revision_before_loading():
    with pytest.raises(ValueError, match="immutable"):
        require_cuda("main")


def test_e5_prefix_mask_pool_and_no_truncation():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    class Inputs(dict):
        def to(self, _):
            return self

    texts = []

    def tokenizer(text, **kwargs):
        texts.append(text)
        assert kwargs["truncation"] is False
        n = 513 if "long" in text else 3
        return Inputs(input_ids=torch.ones((1, n), dtype=torch.long),
                      attention_mask=torch.tensor([[1, 1, 0]]) if n == 3 else torch.ones((1, n)))

    baseline = E5Embedding.__new__(E5Embedding)
    baseline.torch, baseline.tokenizer = torch, tokenizer
    baseline.model = lambda **_: SimpleNamespace(last_hidden_state=torch.tensor([[[2., 0.], [0., 2.], [100., 0.]]]))
    value = baseline.encode(["材料"], role="passage")
    assert texts == ["passage: 材料"]
    np.testing.assert_allclose(value, [[2 ** -0.5, 2 ** -0.5]], atol=1e-6)
    with pytest.raises(BudgetExceeded):
        baseline.encode(["long"])
