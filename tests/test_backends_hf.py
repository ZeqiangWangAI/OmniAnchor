"""No checkpoint downloads: exact scorer checks use tiny causal CPU fixtures."""

import math
import json
from types import SimpleNamespace
import unicodedata

import numpy as np
from PIL import Image
import pytest

from omnianchor.backends.hf import HFBackend, PreparedCandidate, reference_token_logps
from omnianchor.engine import score
from omnianchor.errors import BoundaryError, BudgetExceeded, ResourceUnavailable, OmniAnchorError
from omnianchor.types import Anchor, Bridge, Limits, ModelSpec, Part, ResourceProfile, Sample
from scripts.verify_native import verify_native

torch = pytest.importorskip("torch")


class ByteTokenizer:
    specials = {"<|im_end|>": 256, "<|image_pad|>": 257, "<|video_pad|>": 258}
    all_special_tokens = list(specials)
    all_special_ids = list(specials.values())

    def __init__(self, *, merge_boundary=False, normalize=False):
        self.merge_boundary = merge_boundary
        self.normalize = normalize

    def encode(self, text, add_special_tokens=False):
        if self.normalize:
            text = unicodedata.normalize("NFKC", text)
        result = []
        while text:
            matched = next((token for token in self.specials if text.startswith(token)), None)
            if matched:
                result.append(self.specials[matched])
                text = text[len(matched):]
            elif self.merge_boundary and text.startswith("\nx"):
                result.append(259)
                text = text[2:]
            else:
                result.extend(text[0].encode("utf-8"))
                text = text[1:]
        return result

    def decode(self, ids, **kwargs):
        return bytes(ids).decode("utf-8")

    def convert_tokens_to_ids(self, token):
        return self.specials[token]


class NativeProcessorFixture:
    chat_template = "fixture-template-v1"

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer or ByteTokenizer()
        self.image_processor = SimpleNamespace(patch_size=16, merge_size=2)
        self.video_processor = SimpleNamespace(patch_size=16, temporal_patch_size=2,
                                               cap_pixels_per_frame=True)
        self.calls = []
        self.oversized_grid = False
        self.mutate_pixels = False

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True,
                          "enable_thinking": False}
        pieces = []
        for message in messages:
            if isinstance(message["content"], str):
                pieces.append(message["content"])
            else:
                for part in message["content"]:
                    pieces.append(part["text"] if part["type"] == "text" else
                                  f"<|{part['type']}_pad|>")
        return "U:" + "".join(pieces) + "\nA:"

    def __call__(self, *, text, **kwargs):
        assert kwargs["padding"] is False and kwargs["truncation"] is False
        assert kwargs["add_special_tokens"] is False
        assert kwargs["return_mm_token_type_ids"] is True
        self.calls.append((text[0], kwargs))
        # Expand media tokens before prefix comparison, as a real VLM processor does.
        expanded = text[0].replace("<|image_pad|>", "<|image_pad|>" * 4)
        expanded = expanded.replace("<|video_pad|>", "<|video_pad|>" * 16)
        ids = torch.tensor([self.tokenizer.encode(expanded)], dtype=torch.long)
        mmtypes = torch.where(ids == 257, 1, torch.where(ids == 258, 2, 0))
        result = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                  "mm_token_type_ids": mmtypes}
        for kind, pixels, grid in (("images", "pixel_values", "image_grid_thw"),
                                   ("videos", "pixel_values_videos", "video_grid_thw")):
            if kind in kwargs:
                arrays = [np.asarray(media).astype(float) for media in kwargs[kind]]
                offset = len(self.calls) if self.mutate_pixels else 0
                result[pixels] = torch.tensor([[a.mean() + offset] for a in arrays])
                width = 128 if self.oversized_grid else 2
                result[grid] = torch.tensor([[1 if kind == "images" else 4, width, width]
                                             for _ in arrays])
        return result


class TinyCausalModel(torch.nn.Module):
    """Deterministic full-vocabulary logits depending only on preceding inputs."""

    def __init__(self, *, violate_causality=False):
        super().__init__()
        self.model = SimpleNamespace(rope_deltas="stale")
        self.calls = []
        self.violate_causality = violate_causality

    def forward(self, input_ids, attention_mask, *, logits_to_keep=0, use_cache,
                return_dict, **kwargs):
        assert use_cache is False and return_dict is True
        assert self.model.rope_deltas is None
        self.calls.append({"length": input_ids.shape[1], "selection": logits_to_keep})
        signal = input_ids.float().cumsum(-1)
        for name in ("pixel_values", "pixel_values_videos"):
            if name in kwargs:
                signal = signal + kwargs[name].float().sum()
        if self.violate_causality:
            signal = signal + input_ids.shape[1] * 100
        vocab = torch.arange(260, device=input_ids.device).float()
        logits = torch.sin(signal[..., None] * 0.019 + vocab * 0.37)
        logits += torch.cos(signal[..., None] * 0.007 - vocab * 0.11)
        if isinstance(logits_to_keep, torch.Tensor):
            logits = logits[:, logits_to_keep]
        elif logits_to_keep:
            logits = logits[:, -logits_to_keep:]
        return SimpleNamespace(logits=logits)


def backend(*, processor=None, model=None, resources=None, shared_prefill=False):
    return HFBackend(ModelSpec(device="cpu", precision="fp32"),
                     resources=resources or ResourceProfile(),
                     processor=processor or NativeProcessorFixture(),
                     model=model or TinyCausalModel(), shared_prefill=shared_prefill)


SAMPLE = Sample(id="s", parts=(Part(type="text", text="Evidence"),))
BRIDGE = Bridge(id="b", prefix="Associated with:\n")


def test_reference_shift_mask_and_full_vocabulary_normalization():
    class KnownModel(torch.nn.Module):
        def forward(self, input_ids, *, logits_to_keep, **kwargs):
            probabilities = torch.tensor([[[0.2, 0.6, 0.2], [0.2, 0.3, 0.5],
                                           [0.8, 0.1, 0.1]]])
            return SimpleNamespace(logits=probabilities.log()[:, logits_to_keep])

    ids = torch.tensor([[0, 1, 2]])
    item = PreparedCandidate({"input_ids": ids, "attention_mask": torch.ones_like(ids)},
                             1, Anchor(id="a", surface="ab"), "token_prefix", 2)
    result = reference_token_logps(KnownModel().eval(), item)
    assert result == pytest.approx([math.log(0.6), math.log(0.5)], abs=2e-7)
    assert sum(result) == pytest.approx(math.log(0.3), abs=2e-7)
    item.inputs["attention_mask"][0, 0] = 0
    with pytest.raises(OmniAnchorError, match="Padding"):
        reference_token_logps(KnownModel().eval(), item)


def test_multitoken_chain_unicode_and_turn_end_are_explicit():
    b = backend()
    anchor = Anchor(id="a", surface="公平", language="zh")
    prefix = b.score_candidates(SAMPLE, BRIDGE, [anchor])[0]
    terminated = b.score_candidates(SAMPLE, BRIDGE, [anchor], event="turn_terminated")[0]
    assert prefix["token_ids"] == list("公平".encode())
    assert prefix["raw_logp"] == math.fsum(prefix["token_logps"])
    assert prefix["token_count"] == 6
    assert terminated["token_count"] == 7 and terminated["anchor_token_count"] == 6
    assert terminated["token_ids"] == prefix["token_ids"] + [256]
    assert terminated["token_logps"][:-1] == prefix["token_logps"]
    assert terminated["raw_logp"] < prefix["raw_logp"] < 0
    assert not b._model.training
    assert b._processor.calls[0][0].endswith("A:" + BRIDGE.prefix)


def test_anchor_order_additions_aliases_and_ids_do_not_change_scores():
    b = backend()
    anchors = [Anchor(id="x", surface="x"), Anchor(id="alias", surface="x"),
               Anchor(id="long", surface="freedom")]
    first = {r["anchor_id"]: r["raw_logp"] for r in b.score_candidates(SAMPLE, BRIDGE, anchors)}
    second = {r["anchor_id"]: r["raw_logp"] for r in b.score_candidates(
        SAMPLE.model_copy(update={"id": "new"}), BRIDGE,
        [Anchor(id="extra", surface="equality"), *reversed(anchors)])}
    assert all(first[key] == second[key] for key in first)
    assert first["x"] == first["alias"]
    with pytest.raises(OmniAnchorError, match="Duplicate"):
        b.score_candidates(SAMPLE, BRIDGE, [anchors[0], anchors[0]])


def test_boundary_merge_and_unicode_normalization_fail_closed():
    merged = backend(processor=NativeProcessorFixture(ByteTokenizer(merge_boundary=True))).score_candidates(
        SAMPLE, BRIDGE, [Anchor(id="a", surface="x")])[0]
    assert merged["status"] == "BoundaryError" and "exact prefix" in merged["error"]
    normalized = backend(processor=NativeProcessorFixture(ByteTokenizer(normalize=True))).score_candidates(
        SAMPLE, BRIDGE, [Anchor(id="a", surface="①")])[0]
    assert normalized["status"] == "BoundaryError" and "Unicode" in normalized["error"]


@pytest.mark.parametrize("surface", ["<|im_end|>", "<think>", "x<|image_pad|>y", "\ud800"])
def test_reserved_and_invalid_unicode_anchors_rejected(surface):
    result = backend().score_candidates(SAMPLE, BRIDGE, [Anchor.model_construct(id="a", surface=surface)])[0]
    assert result["status"] == "OmniAnchorError" and result["raw_logp"] is None


@pytest.mark.parametrize("limits, anchor", [
    (Limits(input_text_tokens=2), "a"),
    (Limits(total_postprocessor_tokens=5), "a"),
])
def test_budgets_raise_without_implicit_truncation(limits, anchor):
    with pytest.raises(BudgetExceeded):
        backend(resources=ResourceProfile(limits=limits)).score_candidates(
            SAMPLE, BRIDGE, [Anchor(id="a", surface=anchor)])


@pytest.mark.parametrize("bad", ["ab", "<|im_end|>"])
def test_invalid_candidate_cannot_change_valid_candidate_or_cache_status(tmp_path, bad):
    b = backend(resources=ResourceProfile(limits=Limits(anchor_continuation_tokens=1)))
    good = Anchor(id="good", surface="a")
    invalid = Anchor(id="bad", surface=bad)
    alone = b.score_candidates(SAMPLE, BRIDGE, [good])[0]
    together = b.score_candidates(SAMPLE, BRIDGE, [invalid, good])
    assert together[1] == alone
    assert together[0]["status"] != "ok" and together[0]["raw_logp"] is None
    cold = score([SAMPLE], [invalid, good], [BRIDGE], b)
    score([SAMPLE], [good], [BRIDGE], b, cache=tmp_path)
    warm = score([SAMPLE], [invalid, good], [BRIDGE], b, cache=tmp_path)
    assert cold.frame.status.tolist() == warm.frame.status.tolist()
    assert cold.frame.iloc[1].raw_logp == warm.frame.iloc[1].raw_logp
    assert warm.manifest["execution"]["cache_hits"] == 1


def test_processor_oom_stops_without_scoring_other_candidates():
    processor = NativeProcessorFixture()
    original = processor.__class__.__call__

    class OOMProcessor(NativeProcessorFixture):
        def __call__(self, **kwargs):
            if self.calls:
                raise RuntimeError("CUDA out of memory")
            return original(self, **kwargs)

    b = backend(processor=OOMProcessor())
    with pytest.raises(ResourceUnavailable, match="out of memory"):
        b.score_candidates(SAMPLE, BRIDGE, [Anchor(id="a", surface="a"), Anchor(id="b", surface="b")])
    assert not b._model.calls


def test_media_expansion_grid_budget_and_frozen_pixels(tmp_path):
    path = tmp_path / "tiny.png"
    Image.new("RGB", (50, 40), "red").save(path)
    sample = Sample(id="image", parts=(Part(type="image", path=str(path)),
                                       Part(type="text", text="Observe")))
    b = backend()
    row = b.score_candidates(sample, BRIDGE, [Anchor(id="a", surface="care")])[0]
    assert row["token_count"] == 4
    assert b.last_preparation["grids"]["image_grid_thw"] == [[1, 2, 2]]
    assert b.last_preparation["media"][0]["width"] == 50
    assert b._processor.calls[0][1]["images_kwargs"]["size"]["longest_edge"] == 262144
    pure_text = backend().score_candidates(SAMPLE, BRIDGE, [Anchor(id="a", surface="care")])[0]
    assert row["raw_logp"] != pure_text["raw_logp"]
    processor = NativeProcessorFixture()
    processor.oversized_grid = True
    with pytest.raises(BudgetExceeded, match="actual pixel"):
        backend(processor=processor).score_candidates(sample, BRIDGE, [Anchor(id="a", surface="a")])
    processor.oversized_grid = False
    processor.mutate_pixels = True
    with pytest.raises(BoundaryError, match="frozen media"):
        backend(processor=processor).score_candidates(sample, BRIDGE, [Anchor(id="a", surface="a")])


def test_real_video_decode_reaches_native_processor_and_teacher_forcing(tmp_path):
    av = pytest.importorskip("av")
    path = tmp_path / "video.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=4)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for index in range(8):
            pixels = np.full((32, 32, 3), index * 30, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    b = backend()
    sample = Sample(id="v", parts=(Part(type="video", path=str(path)),))
    row = b.score_candidates(sample, BRIDGE, [Anchor(id="a", surface="motion")])[0]
    assert row["token_count"] == 6 and math.isfinite(row["raw_logp"])
    assert b.last_preparation["grids"]["video_grid_thw"] == [[4, 2, 2]]
    native = b._processor.calls[0][1]["videos_kwargs"]
    assert native["video_metadata"][0]["fps"] == 4
    assert native["video_metadata"][0]["frames_indices"] == list(range(8))
    assert native["do_sample_frames"] is False
    assert native["cap_pixels_per_frame"] is False


def test_shared_single_token_prefill_requires_runtime_validation_and_saves_calls():
    b = backend(shared_prefill=True)
    anchors = [Anchor(id="a", surface="a"), Anchor(id="b", surface="b")]
    initial = b.score_candidates(SAMPLE, BRIDGE, anchors)
    assert b.shared_prefill_validation["passed"]
    assert len(b._model.calls) == 3  # Shared prefix plus both independent references.
    repeated = b.score_candidates(SAMPLE, BRIDGE, anchors)
    assert initial == repeated and len(b._model.calls) == 4
    assert b.identity["multi_token_cache"] is False
    assert backend().shared_prefill is False


def test_failed_shared_prefill_validation_returns_reference_and_disables_fast_path():
    b = backend(shared_prefill=True, model=TinyCausalModel(violate_causality=True))
    anchors = [Anchor(id="a", surface="a"), Anchor(id="b", surface="b")]
    first = b.score_candidates(SAMPLE, BRIDGE, anchors)
    assert b.shared_prefill_validation["passed"] is False
    assert len(b._model.calls) == 3
    second = b.score_candidates(SAMPLE, BRIDGE, anchors)
    assert first == second and len(b._model.calls) == 5


def test_real_model_cpu_refused_before_any_transformers_load():
    b = HFBackend(ModelSpec(device="cpu"))
    with pytest.raises(ResourceUnavailable, match="no checkpoint was downloaded"):
        b.score_candidates(SAMPLE, BRIDGE, [Anchor(id="a", surface="care")])
    assert b._model is None and b._processor is None


def test_native_acceptance_compares_selected_and_full_logits_and_writes_json(tmp_path):
    b = backend()
    target = tmp_path / "verification.json"
    anchors = [Anchor(id="a", surface="a"), Anchor(id="long", surface="bc")]
    report = verify_native(b, output=target, sample=SAMPLE, bridge=BRIDGE, anchors=anchors)
    assert report["status"] == "passed"
    full = report["checks"]["selected_vs_full"]
    assert full["normalization_dtype"] == "float32"
    assert [r["token_count"] for r in full["candidates"]] == [1, 2]
    assert all(r["max_absolute_token_logp_error"] <= 0.01 for r in full["candidates"])
    assert report["checks"]["shared_prefill"]["passed"] is True
    assert any(isinstance(call["selection"], int) and call["selection"] == 0 for call in b._model.calls)
    assert json.loads(target.read_text())["checks"]["selected_vs_full"] == full


def test_native_acceptance_detects_wrong_selected_logits():
    class BrokenSelection(TinyCausalModel):
        def forward(self, *args, **kwargs):
            result = super().forward(*args, **kwargs)
            if isinstance(kwargs["logits_to_keep"], torch.Tensor):
                result.logits[..., 97] += 1.0
            return result

    report = verify_native(backend(model=BrokenSelection()), anchors=[Anchor(id="a", surface="a")])
    assert report["status"] == "failed"
    assert report["checks"]["selected_vs_full"]["passed"] is False


def test_native_acceptance_treats_disabled_prefill_as_safe_reference_fallback():
    b = backend(model=TinyCausalModel(violate_causality=True), shared_prefill=True)
    report = verify_native(b, anchors=[Anchor(id="a", surface="a")])
    shared = report["checks"]["shared_prefill"]
    assert report["status"] == "passed"
    assert shared["optimization_status"] == "disabled"
    assert shared["fallback"] == "uncached_reference"
    assert shared["runtime_validation"]["max_absolute_token_logp_error"] > 0.01
    assert b.shared_prefill_validation["passed"] is False
    reused = verify_native(b, anchors=[Anchor(id="a", surface="a")],
                           existing_checks={"shared_prefill": b.shared_prefill_validation})
    assert reused["status"] == "passed"
    assert reused["checks"]["shared_prefill"]["optimization_status"] == "disabled"


def test_native_acceptance_checks_image_then_text_and_can_reuse_live_evidence(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (40, 50), "blue").save(path)
    image_sample = Sample(id="i", parts=(Part(type="image", path=str(path)),))
    b = backend(shared_prefill=True)
    anchors = [Anchor(id="a", surface="a")]
    report = verify_native(b, anchors=anchors, image_sample=image_sample)
    state = report["checks"]["modality_state"]
    assert state["passed"] and state["order"] == ["text", "image", "text"]
    assert state["comparisons"][0]["grid"] == [[1, 2, 2]]
    assert b.shared_prefill is True
    reused = verify_native(b, anchors=anchors, existing_checks={
        "shared_prefill": report["checks"]["shared_prefill"], "modality_state": state,
    })
    assert reused["checks"]["modality_state"]["source"] == "reused_live_demo_evidence"
    assert reused["checks"]["shared_prefill"]["source"] == "reused_live_demo_evidence"


def test_native_acceptance_refuses_to_load_a_model_or_materialize_large_full_logits():
    b = HFBackend(ModelSpec(device="cpu"))
    with pytest.raises(ResourceUnavailable, match="already-loaded"):
        verify_native(b)
    assert b._model is None and b._processor is None
    b = backend()
    with pytest.raises(BudgetExceeded, match="full-logit memory bound"):
        verify_native(b, max_full_sequence_tokens=2)
    assert not b._model.calls
