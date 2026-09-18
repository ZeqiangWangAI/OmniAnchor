"""Model adapters select the scoring path per model family without touching the math."""

import pytest

from omnianchor.backends.adapters import (ADAPTERS, TextOnlyProcessor, adapter_for_model_id,
                                          resolve_adapter)
from omnianchor.backends.hf import HFBackend
from omnianchor.engine import score
from omnianchor.errors import OmniAnchorError
from omnianchor.types import Anchor, Bridge, ModelSpec, Part, Sample

torch = pytest.importorskip("torch")

from test_backends_hf import ByteTokenizer, NativeProcessorFixture, TinyCausalModel  # noqa: E402

SAMPLE = Sample(id="s", parts=(Part(type="text", text="Evidence"),))
BRIDGE = Bridge(id="b", prefix="Associated with:\n")


class ChatTokenizer(ByteTokenizer):
    """Byte tokenizer that also renders a chat template, as a text-only HF tokenizer does."""

    chat_template = "text-template-v1"

    def __init__(self):
        super().__init__()
        self.template_kwargs = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_kwargs.append(kwargs)
        assert all(isinstance(m["content"], str) for m in messages)
        return "U:" + "".join(m["content"] for m in messages) + "\nA:"

    def __call__(self, text, *, return_tensors, padding, truncation, add_special_tokens):
        assert return_tensors == "pt" and not padding and not truncation and not add_special_tokens
        ids = torch.tensor([self.encode(text[0])], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def text_backend(spec: ModelSpec, tokenizer=None):
    return HFBackend(spec, processor=TextOnlyProcessor(tokenizer or ChatTokenizer()),
                     model=TinyCausalModel())


def test_registry_covers_documented_families():
    assert {"qwen3_5", "qwen3_vl", "qwen3", "causal_lm"} <= set(ADAPTERS)
    assert ADAPTERS["qwen3_5"].modalities == frozenset({"text", "image", "video"})
    assert ADAPTERS["causal_lm"].modalities == frozenset({"text"})
    assert ADAPTERS["causal_lm"].turn_end_token is None


def test_resolution_prefers_explicit_then_model_type_then_registry_id():
    assert resolve_adapter(ModelSpec(adapter="causal_lm"), model_type="qwen3_5").name == "causal_lm"
    assert resolve_adapter(ModelSpec(id="any/model"), model_type="qwen3_vl").name == "qwen3_vl"
    assert resolve_adapter(ModelSpec(id="Qwen/Qwen3-VL-4B-Instruct")).name == "qwen3_vl"
    assert resolve_adapter(ModelSpec(id="HuggingFaceTB/SmolLM3-3B"), model_type="smollm3").name == "causal_lm"
    assert adapter_for_model_id("Qwen/Qwen3.5-4B") == "qwen3_5"
    with pytest.raises(OmniAnchorError, match="No adapter"):
        resolve_adapter(ModelSpec(id="unknown/model"), model_type="unknown_arch")
    with pytest.raises(OmniAnchorError, match="Unknown adapter"):
        resolve_adapter(ModelSpec(adapter="nope"))


def test_default_backend_keeps_the_qwen_native_path_and_records_the_adapter():
    b = HFBackend(ModelSpec(device="cpu", precision="fp32"),
                  processor=NativeProcessorFixture(), model=TinyCausalModel())
    assert b.adapter.name == "qwen3_5" and b.identity["adapter"] == "qwen3_5"
    assert b.identity["adapter_kind"] == "native_multimodal"
    assert b.identity["chat_template_kwargs"] == {"enable_thinking": False}
    assert b.identity["turn_end_token"] == "<|im_end|>"
    row = b.score_candidates(SAMPLE, BRIDGE, [Anchor(id="a", surface="a")])[0]
    assert row["status"] == "ok"


def test_text_only_adapter_scores_text_without_multimodal_kwargs():
    tokenizer = ChatTokenizer()
    b = text_backend(ModelSpec(id="HuggingFaceTB/SmolLM3-3B", adapter="causal_lm", device="cpu",
                               precision="fp32", turn_end_token="<|im_end|>"), tokenizer)
    rows = b.score_candidates(SAMPLE, BRIDGE, [Anchor(id="a", surface="ab")])
    assert rows[0]["status"] == "ok" and rows[0]["token_count"] == 2
    assert tokenizer.template_kwargs == [{"tokenize": False, "add_generation_prompt": True}]
    assert b.identity["adapter_kind"] == "causal_lm"
    terminated = b.score_candidates(SAMPLE, BRIDGE, [Anchor(id="a", surface="ab")],
                                    event="turn_terminated")[0]
    assert terminated["token_count"] == 3 and terminated["token_ids"][-1] == 256


def test_qwen3_text_adapter_disables_thinking_in_the_template():
    tokenizer = ChatTokenizer()
    b = text_backend(ModelSpec(id="Qwen/Qwen3-4B", adapter="qwen3", device="cpu", precision="fp32"),
                     tokenizer)
    b.score_candidates(SAMPLE, BRIDGE, [Anchor(id="a", surface="a")])
    assert tokenizer.template_kwargs[0]["enable_thinking"] is False
    assert b.identity["turn_end_token"] == "<|im_end|>"


def test_explicit_chat_template_kwargs_override_the_adapter_default():
    tokenizer = ChatTokenizer()
    b = text_backend(ModelSpec(id="x/y", adapter="causal_lm", device="cpu", precision="fp32",
                               chat_template_kwargs={"enable_thinking": False}), tokenizer)
    b.score_candidates(SAMPLE, BRIDGE, [Anchor(id="a", surface="a")])
    assert tokenizer.template_kwargs[0]["enable_thinking"] is False


def test_generic_adapter_never_guesses_a_turn_end_token():
    b = text_backend(ModelSpec(id="x/y", adapter="causal_lm", device="cpu", precision="fp32"))
    rows = b.score_candidates(SAMPLE, BRIDGE, [Anchor(id="a", surface="a")], event="turn_terminated")
    assert rows[0]["status"] == "OmniAnchorError" and "turn_end_token" in rows[0]["error"]
    assert b.score_candidates(SAMPLE, BRIDGE, [Anchor(id="a", surface="a")])[0]["status"] == "ok"


def test_text_only_adapter_fails_media_samples_explicitly(tmp_path):
    from PIL import Image
    path = tmp_path / "i.png"
    Image.new("RGB", (8, 8), "red").save(path)
    media = Sample(id="m", parts=(Part(type="image", path=str(path)),))
    b = text_backend(ModelSpec(id="x/y", adapter="causal_lm", device="cpu", precision="fp32"))
    with pytest.raises(OmniAnchorError, match="accepts text only"):
        b.score_candidates(media, BRIDGE, [Anchor(id="a", surface="a")])
    table = score([media, SAMPLE], [Anchor(id="a", surface="a")], [BRIDGE], b)
    assert table.frame.set_index("sample_id").status.to_dict() == {"m": "OmniAnchorError", "s": "ok"}
    assert table.manifest["model"]["adapter"] == "causal_lm"


def test_unknown_model_without_adapter_is_refused_before_any_download():
    b = HFBackend(ModelSpec(id="unknown/model", device="cpu"))
    with pytest.raises(OmniAnchorError):
        b.score_candidates(SAMPLE, BRIDGE, [Anchor(id="a", surface="a")])
    assert b._model is None
