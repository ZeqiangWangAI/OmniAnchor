from types import SimpleNamespace

import pytest

from vlanchor.backends.vision_reuse import reuse_vision_outputs


def test_vision_reuse_checks_inputs_isolates_mutation_and_restores_after_error():
    torch = pytest.importorskip("torch")

    class Visual:
        def __init__(self):
            self.count = 0

        def get_image_features(self, values, grid):
            self.count += 1
            return {"features": values*2, "grid": grid}

    visual = Visual()
    model = SimpleNamespace(training=False, model=visual)
    values, grid = torch.arange(4.), torch.tensor([1, 2, 2])
    with pytest.raises(ValueError, match="inputs changed"):
        with reuse_vision_outputs(model) as stats:
            first = visual.get_image_features(values, grid)
            first["features"].zero_()
            second = visual.get_image_features(values.clone(), grid.clone())
            assert torch.equal(second["features"], values*2)
            assert visual.count == 1
            assert stats["calls"]["get_image_features"] == 2
            visual.get_image_features(values+1, grid)
    assert "get_image_features" not in visual.__dict__
    visual.get_image_features(values, grid)
    assert visual.count == 2
    with reuse_vision_outputs(model):
        visual.get_image_features(values, grid)
    assert visual.count == 3  # A new scope recomputes, never reuses another material.
