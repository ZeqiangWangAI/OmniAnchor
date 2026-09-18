"""Optional exact vision-output reuse within one candidate call; language KV remains uncached."""
from contextlib import contextmanager
from copy import deepcopy

from .hf import HFBackend


def _same_inputs(left, right):
    import torch
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.dtype == right.dtype and left.device == right.device
                and left.shape == right.shape and torch.equal(left, right))
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same_inputs(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(_same_inputs(a, b) for a, b in zip(left, right))
    return left == right


@contextmanager
def reuse_vision_outputs(model):
    """Cache only independent visual encoder outputs; reject any input change and restore hooks.

    Scope is exactly one score_candidates/paired-query call. Inputs are compared by
    dtype, device, shape and exact tensor equality. Copies isolate cached outputs
    from any downstream mutation. No decoder state, logits or KV tensors are cached.
    """
    if model.training:
        raise ValueError("Vision reuse requires eval mode.")
    base = getattr(model, "model", model)
    names = [name for name in ["get_image_features", "get_video_features"] if hasattr(base, name)]
    if not names:
        raise ValueError("No supported independent visual feature methods.")
    originals, stats = {}, {"calls": {}, "computations": {}, "scope": "one_candidate_call"}

    def wrap(name, original):
        cached = None

        def invoke(*args, **kwargs):
            nonlocal cached
            stats["calls"][name] = stats["calls"].get(name, 0)+1
            if cached is None:
                result = original(*args, **kwargs)
                cached = (deepcopy((args, kwargs)), deepcopy(result))
                stats["computations"][name] = stats["computations"].get(name, 0)+1
                return result
            if not _same_inputs(cached[0], (args, kwargs)):
                raise ValueError("Visual inputs changed within a cached candidate call.")
            return deepcopy(cached[1])
        return invoke

    try:
        for name in names:
            original = getattr(base, name)
            originals[name] = (name in base.__dict__, base.__dict__.get(name))
            setattr(base, name, wrap(name, original))
        yield stats
    finally:
        for name, (was_instance_attribute, value) in originals.items():
            if was_instance_attribute:
                setattr(base, name, value)
            else:
                delattr(base, name)


class VisionReuseBackend(HFBackend):
    def __init__(self, *args, reuse_vision_features=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.reuse_vision_features = reuse_vision_features
        self.last_vision_reuse = {}

    @property
    def identity(self):
        base = super().identity
        if self.reuse_vision_features:
            base["vision_feature_reuse"] = "exact_inputs_copied_outputs_one_candidate_call_v1"
        return base

    def score_candidates(self, *args, **kwargs):
        if not self.reuse_vision_features:
            return super().score_candidates(*args, **kwargs)
        self._ensure_loaded()
        with reuse_vision_outputs(self._model) as stats:
            result = super().score_candidates(*args, **kwargs)
        self.last_vision_reuse = stats
        return result
