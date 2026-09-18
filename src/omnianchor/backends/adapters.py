"""Model-family adapters: what differs between checkpoints, kept apart from the scoring math.

An adapter names the Transformers class, the modalities the processor accepts, the
chat-template keyword arguments, and the exact turn-end token. The strict encoding,
prefix-preservation and teacher-forcing checks in ``hf.py`` are identical for every
adapter; only these declarations change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..errors import OmniAnchorError
from ..types import ModelSpec


@dataclass(frozen=True)
class ModelAdapter:
    name: str
    kind: str  # "native_multimodal" or "causal_lm"
    model_types: tuple[str, ...]
    model_class: str
    modalities: frozenset[str]
    chat_template_kwargs: Mapping[str, Any] = field(default_factory=dict)
    turn_end_token: str | None = None
    requires_mm_token_type_ids: bool = False

    @property
    def text_only(self) -> bool:
        return self.modalities == frozenset({"text"})


ADAPTERS: dict[str, ModelAdapter] = {
    "qwen3_5": ModelAdapter(
        name="qwen3_5", kind="native_multimodal", model_types=("qwen3_5",),
        model_class="Qwen3_5ForConditionalGeneration",
        modalities=frozenset({"text", "image", "video"}),
        chat_template_kwargs={"enable_thinking": False}, turn_end_token="<|im_end|>",
        requires_mm_token_type_ids=True,
    ),
    "qwen3_vl": ModelAdapter(
        name="qwen3_vl", kind="native_multimodal", model_types=("qwen3_vl",),
        model_class="Qwen3VLForConditionalGeneration",
        modalities=frozenset({"text", "image", "video"}),
        chat_template_kwargs={"enable_thinking": False}, turn_end_token="<|im_end|>",
        requires_mm_token_type_ids=True,
    ),
    "qwen3": ModelAdapter(
        name="qwen3", kind="causal_lm", model_types=("qwen3",),
        model_class="AutoModelForCausalLM", modalities=frozenset({"text"}),
        chat_template_kwargs={"enable_thinking": False}, turn_end_token="<|im_end|>",
    ),
    "causal_lm": ModelAdapter(
        name="causal_lm", kind="causal_lm", model_types=(),
        model_class="AutoModelForCausalLM", modalities=frozenset({"text"}),
    ),
}


def adapter_for_model_id(model_id: str) -> str | None:
    """Adapter name recorded for a pinned model ID in the packaged registry, if any."""
    from ..config import MODEL_REGISTRY
    for entry in MODEL_REGISTRY.values():
        if entry.get("id") == model_id and entry.get("backend") == "hf":
            return entry.get("adapter")
    return None


def resolve_adapter(spec: ModelSpec, model_type: str | None = None) -> ModelAdapter:
    """Explicit ``spec.adapter`` wins, then the checkpoint's ``model_type``, then the registry."""
    if spec.adapter is not None:
        try:
            return ADAPTERS[spec.adapter]
        except KeyError as exc:
            raise OmniAnchorError(f"Unknown adapter {spec.adapter!r}; choose one of "
                                  f"{', '.join(sorted(ADAPTERS))}.") from exc
    if model_type is not None:
        for adapter in ADAPTERS.values():
            if model_type in adapter.model_types:
                return adapter
    name = adapter_for_model_id(spec.id)
    if name is not None:
        return ADAPTERS[name]
    raise OmniAnchorError(
        f"No adapter for {spec.id!r}" + (f" (model_type {model_type!r})" if model_type else "")
        + "; set model.adapter explicitly (text-only chat models can use 'causal_lm')."
    )


class TextOnlyProcessor:
    """Present a plain tokenizer through the processor interface used by the strict scorer."""

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    @property
    def chat_template(self) -> Any:
        return getattr(self.tokenizer, "chat_template", None)

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        flat = []
        for message in messages:
            content = message["content"]
            if not isinstance(content, str):
                if any(part.get("type") != "text" for part in content):
                    raise OmniAnchorError("A text-only adapter accepts text only.")
                content = "".join(part["text"] for part in content)
            flat.append({"role": message["role"], "content": content})
        return self.tokenizer.apply_chat_template(flat, **kwargs)

    def __call__(self, *, text: list[str], return_tensors: str, padding: bool, truncation: bool,
                 add_special_tokens: bool, return_mm_token_type_ids: bool = False,
                 **media: Any) -> dict[str, Any]:
        if media:
            raise OmniAnchorError("A text-only adapter accepts text only.")
        encoded = self.tokenizer(text, return_tensors=return_tensors, padding=padding,
                                 truncation=truncation, add_special_tokens=add_special_tokens)
        return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}
