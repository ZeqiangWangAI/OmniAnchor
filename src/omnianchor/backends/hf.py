"""Strict native teacher forcing, with an unpadded, uncached reference path.

Model-family differences (class, modalities, template arguments, turn-end token) come
from ``adapters.py``; the encoding and scoring checks below are the same for every model.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
import math
import re
from typing import Any, Sequence
from urllib.parse import urlsplit

from ..errors import BoundaryError, BudgetExceeded, ResourceUnavailable, OmniAnchorError
from ..types import Anchor, Bridge, Event, ModelSpec, ResourceProfile, Sample
from .adapters import ModelAdapter, TextOnlyProcessor, resolve_adapter
from .media import FrozenMedia, freeze_media, processor_media_kwargs


_MODEL_INPUTS = {
    "input_ids", "attention_mask", "mm_token_type_ids", "pixel_values",
    "pixel_values_videos", "image_grid_thw", "video_grid_thw",
}


class _MediaPrefixError(BoundaryError):
    """Frozen input mutation invalidates the sample, not one anchor surface."""


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ResourceUnavailable("HF scoring requires torch: install omnianchor[hf].") from exc
    return torch


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _transformers_source_revision() -> str | None:
    """Expose only an immutable commit, never installation URLs or credentials."""
    try:
        raw = metadata.distribution("transformers").read_text("direct_url.json")
        direct = json.loads(raw or "{}")
        if not isinstance(direct, dict):
            return None
        vcs = direct.get("vcs_info")
        revision = vcs.get("commit_id") if isinstance(vcs, dict) else None
        if isinstance(revision, str) and re.fullmatch(r"[a-fA-F0-9]{40}", revision):
            return revision.lower()
        source = direct.get("url")
        if not isinstance(source, str):
            return None
        parsed = urlsplit(source)
        if parsed.scheme != "https":
            return None
        pattern = {
            "github.com": r"/huggingface/transformers/archive/([a-fA-F0-9]{40})\.(?:tar\.gz|zip)",
            "codeload.github.com": r"/huggingface/transformers/(?:tar\.gz|zip)/([a-fA-F0-9]{40})",
        }.get(parsed.hostname)
        match = re.fullmatch(pattern, parsed.path) if pattern else None
        return match.group(1).lower() if match else None
    except (metadata.PackageNotFoundError, OSError, ValueError, TypeError):
        return None


@dataclass
class PreparedCandidate:
    inputs: dict[str, Any]
    prefix_length: int
    anchor: Anchor
    event: Event
    anchor_token_count: int


def reference_token_logps(model: Any, item: PreparedCandidate) -> list[float]:
    """One complete, unpadded forward; retain only positions predicting the target."""
    torch = _torch()
    ids = item.inputs["input_ids"]
    attention = item.inputs["attention_mask"]
    if getattr(model, "training", False):
        raise OmniAnchorError("Scoring requires model.eval().")
    if ids.ndim != 2 or ids.shape[0] != 1 or attention.shape != ids.shape:
        raise OmniAnchorError("Reference scoring requires one unpadded sequence.")
    if not bool(attention.eq(1).all()):
        raise OmniAnchorError("Padding is not supported by the reference scorer.")
    p, length = item.prefix_length, ids.shape[1]
    if not 1 <= p < length:
        raise BoundaryError("Invalid prefix or empty candidate continuation.")
    if set(item.inputs) - _MODEL_INPUTS:
        raise OmniAnchorError("Unexpected model inputs in reference scoring.")
    target_positions = torch.arange(p, length, device=ids.device)
    # Some Transformers versions keep M-RoPE state on the model, outside Cache.
    base = getattr(model, "model", None)
    if base is not None and hasattr(base, "rope_deltas"):
        base.rope_deltas = None
    with torch.inference_mode():
        out = model(**item.inputs, use_cache=False, return_dict=True,
                    logits_to_keep=target_positions - 1)
        logits = out.logits[0].float()
        if logits.ndim != 2 or logits.shape[0] != target_positions.numel():
            raise OmniAnchorError("Backend did not honor selected-logit positions.")
        targets = ids[0, target_positions]
        logps = logits.log_softmax(-1).gather(-1, targets[:, None]).squeeze(-1)
        if not bool(torch.isfinite(logps).all()):
            raise OmniAnchorError("Non-finite anchor log probabilities.")
        return [float(value) for value in logps.cpu().tolist()]


class HFBackend:
    """Load real checkpoints only on explicitly available CUDA resources.

    Injected model/processor pairs permit small CPU unit tests without downloads.
    Shared prefill is opt-in and checked against every single-token candidate on
    its first use in this backend instance. Hybrid cache branching is not enabled.
    """

    def __init__(
        self, model_spec: ModelSpec, resources: ResourceProfile = ResourceProfile(),
        system_prompt: str | None = None, *, model: Any = None, processor: Any = None,
        shared_prefill: bool = False,
    ):
        if (model is None) != (processor is None):
            raise OmniAnchorError("Inject both model and processor, or neither.")
        self.model_spec = model_spec
        self.resources = resources
        self.system_prompt = system_prompt
        self._model = model
        self._processor = processor
        self._injected = model is not None
        self.shared_prefill = shared_prefill
        self.shared_prefill_validation: dict[str, Any] | None = None
        self.last_preparation: dict[str, Any] = {}
        # Without a loaded checkpoint the adapter can only come from the explicit field or
        # the packaged registry; otherwise it is resolved from config.model_type at load.
        self.adapter: ModelAdapter | None = None
        try:
            self.adapter = resolve_adapter(model_spec)
        except OmniAnchorError:
            if self._injected:
                raise
        if self._injected:
            self._model.eval()

    @property
    def chat_template_kwargs(self) -> dict[str, Any]:
        if self.model_spec.chat_template_kwargs is not None:
            return dict(self.model_spec.chat_template_kwargs)
        return dict(self.adapter.chat_template_kwargs) if self.adapter else {}

    @property
    def turn_end_token(self) -> str | None:
        if self.model_spec.turn_end_token is not None:
            return self.model_spec.turn_end_token
        return self.adapter.turn_end_token if self.adapter else None

    @property
    def identity(self) -> dict[str, Any]:
        result = {
            "backend": "hf", **self.model_spec.model_dump(),
            "adapter": self.adapter.name if self.adapter else None,
            "adapter_kind": self.adapter.kind if self.adapter else None,
            "scoring_policy": "native-strict-v1", "injected_test_runtime": self._injected,
            "resources": self.resources.model_dump(), "system_prompt": self.system_prompt,
            "shared_prefill_requested": self.shared_prefill,
            "multi_token_cache": False, "chat_template_kwargs": self.chat_template_kwargs,
            "turn_end_token": self.turn_end_token,
            "torch_version": _version("torch"), "transformers_version": _version("transformers"),
            "transformers_source_revision": _transformers_source_revision(),
            "processor_revision": self.model_spec.revision,
            "tokenizers_version": _version("tokenizers"), "pillow_version": _version("Pillow"),
            "causal_conv1d_version": _version("causal-conv1d"),
            "flash_linear_attention_version": _version("flash-linear-attention"),
            "av_version": _version("av"),
        }
        return result

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        torch = _torch()
        spec = self.model_spec
        if not re.fullmatch(r"[a-fA-F0-9]{40}", spec.revision):
            raise OmniAnchorError("Real model revisions must be immutable 40-character commit hashes.")
        try:
            device = torch.device(spec.device)
        except (RuntimeError, ValueError) as exc:
            raise ResourceUnavailable(f"Invalid model device: {spec.device}") from exc
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ResourceUnavailable(
                "Production HF scoring requires an explicitly available CUDA device; "
                "no checkpoint was downloaded and no CPU fallback was attempted."
            )
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise ResourceUnavailable(f"CUDA device is outside the visible allocation: {device}")
        with torch.cuda.device(index):
            if spec.precision == "bf16" and not torch.cuda.is_bf16_supported():
                raise ResourceUnavailable("Requested CUDA allocation does not support BF16.")
        try:
            import transformers
        except ImportError as exc:
            raise ResourceUnavailable("HF scoring requires transformers: install omnianchor[hf].") from exc
        try:
            config = transformers.AutoConfig.from_pretrained(spec.id, revision=spec.revision)
        except Exception as exc:
            raise ResourceUnavailable(f"Cannot read the checkpoint configuration: {exc}") from exc
        adapter = resolve_adapter(spec, model_type=getattr(config, "model_type", None))
        cls = getattr(transformers, adapter.model_class, None)
        if cls is None:
            raise ResourceUnavailable(
                f"Installed Transformers lacks {adapter.model_class} for adapter {adapter.name}."
            )
        kwargs: dict[str, Any] = {
            "revision": spec.revision, "dtype": torch.bfloat16 if spec.precision == "bf16"
            else torch.float32, "device_map": {"": f"cuda:{index}"},
        }
        if spec.attention_implementation:
            kwargs["attn_implementation"] = spec.attention_implementation
        try:
            if adapter.text_only:
                self._processor = TextOnlyProcessor(transformers.AutoTokenizer.from_pretrained(
                    spec.id, revision=spec.revision,
                ))
            else:
                self._processor = transformers.AutoProcessor.from_pretrained(
                    spec.id, revision=spec.revision,
                )
            self._model = cls.from_pretrained(spec.id, **kwargs).eval()
        except Exception as exc:
            self._model = None
            raise ResourceUnavailable(f"Cannot load the requested CUDA checkpoint: {exc}") from exc
        self.adapter = adapter

    def _encode(self, text: str, frozen: FrozenMedia) -> dict[str, Any]:
        torch = _torch()
        processor = self._processor
        video_processor = getattr(processor, "video_processor", None)
        media_kwargs = processor_media_kwargs(
            frozen, self.resources,
            temporal_patch_size=getattr(video_processor, "temporal_patch_size", 2),
            cap_pixels_supported=hasattr(video_processor, "cap_pixels_per_frame"),
        )
        try:
            encoded = processor(
                text=[text], return_tensors="pt", padding=False, truncation=False,
                add_special_tokens=False, return_mm_token_type_ids=True, **media_kwargs,
            )
        except RuntimeError as exc:
            raise ResourceUnavailable(f"Native processor runtime failed: {exc}") from exc
        except Exception as exc:
            raise OmniAnchorError(f"Native processor failed: {exc}") from exc
        inputs = {name: encoded[name] for name in _MODEL_INPUTS if name in encoded}
        ids = inputs.get("input_ids")
        attention = inputs.get("attention_mask")
        if ids is None or attention is None or ids.ndim != 2 or ids.shape[0] != 1:
            raise OmniAnchorError("Processor must return one sequence with an attention mask.")
        if attention.shape != ids.shape or not bool(attention.eq(1).all()):
            raise OmniAnchorError("Processor unexpectedly padded or masked the sequence.")
        if ids.shape[1] > self.resources.limits.total_postprocessor_tokens:
            raise BudgetExceeded("Full processor-expanded sequence exceeds its token budget.")
        if (frozen.images or frozen.videos) and self.adapter.requires_mm_token_type_ids:
            if "mm_token_type_ids" not in inputs or inputs["mm_token_type_ids"].shape != ids.shape:
                raise OmniAnchorError("Native multimodal scoring requires matching mm_token_type_ids.")
        for kind, pixel_key, grid_key, max_pixels in (
            ("image", "pixel_values", "image_grid_thw", self.resources.image.max_pixels),
            ("video", "pixel_values_videos", "video_grid_thw",
             self.resources.video.max_pixels_per_frame),
        ):
            media = frozen.images if kind == "image" else frozen.videos
            if not media:
                continue
            if pixel_key not in inputs or grid_key not in inputs:
                raise OmniAnchorError(f"Processor dropped {kind} tensors or spatial grids.")
            native = processor.image_processor if kind == "image" else video_processor
            patch = getattr(native, "patch_size", None)
            if not isinstance(patch, int) or patch <= 0:
                raise OmniAnchorError("Native processor must declare its actual spatial patch size.")
            grid = inputs[grid_key]
            if grid.ndim != 2 or grid.shape != (len(media), 3) or not bool(grid.gt(0).all()):
                raise OmniAnchorError(f"Invalid {kind} grid shape or media count.")
            areas = grid[:, 1] * grid[:, 2] * patch * patch
            if bool(areas.gt(max_pixels).any()):
                raise BudgetExceeded(f"Native {kind} resizing exceeds the actual pixel budget.")
        if not all(isinstance(value, torch.Tensor) for value in inputs.values()):
            raise OmniAnchorError("Native model inputs must be tensors.")
        return inputs

    def _device_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        torch = _torch()
        dtype = torch.bfloat16 if self.model_spec.precision == "bf16" else torch.float32
        return {
            name: value.to(device=self.model_spec.device, dtype=dtype)
            if name in ("pixel_values", "pixel_values_videos")
            else value.to(device=self.model_spec.device)
            for name, value in inputs.items()
        }

    def _check_prefix(self, prefix: dict[str, Any], full: dict[str, Any]) -> int:
        torch = _torch()
        n = prefix["input_ids"].shape[1]
        if full["input_ids"].shape[1] <= n or not torch.equal(
            prefix["input_ids"], full["input_ids"][:, :n],
        ):
            raise BoundaryError(
                "Full tokenization does not preserve the exact prefix. Change the explicit "
                "bridge delimiter; independently tokenized anchors will not be concatenated."
            )
        for name in ("mm_token_type_ids", "attention_mask"):
            if name in prefix and (name not in full or not torch.equal(prefix[name], full[name][:, :n])):
                raise BoundaryError(f"Processor changed prefix {name}.")
        for name in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"):
            if (name in prefix) != (name in full) or (
                name in prefix and not torch.equal(prefix[name], full[name])
            ):
                raise _MediaPrefixError(f"Processor changed frozen media field {name}.")
        return n

    def render_prefix(self, frozen: FrozenMedia, bridge: Bridge) -> str:
        """Official chat template (user turn, generation prompt) followed by the exact bridge."""
        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": frozen.content})
        return self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **self.chat_template_kwargs,
        ) + bridge.prefix

    def _validate_surface(self, surface: str) -> None:
        special = getattr(self._processor.tokenizer, "all_special_tokens", [])
        if any(token and token in surface for token in special) or any(
            token in surface for token in ("<|", "<think>", "</think>", "<tool_call>")
        ):
            raise OmniAnchorError("Anchor contains a model control token.")
        try:
            surface.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise OmniAnchorError("Anchor contains invalid Unicode surrogate code points.") from exc

    def _compile(
        self, prefix_text: str, prefix: dict[str, Any], frozen: FrozenMedia,
        anchor: Anchor, event: Event,
    ) -> PreparedCandidate:
        torch = _torch()
        tokenizer = self._processor.tokenizer
        self._validate_surface(anchor.surface)
        full = self._encode(prefix_text + anchor.surface, frozen)
        n = self._check_prefix(prefix, full)
        continuation = full["input_ids"][0, n:].tolist()
        if any(token in set(getattr(tokenizer, "all_special_ids", [])) for token in continuation):
            raise OmniAnchorError("Anchor continuation includes reserved model tokens.")
        if tokenizer.decode(continuation, skip_special_tokens=False,
                            clean_up_tokenization_spaces=False) != anchor.surface:
            raise BoundaryError("Tokenizer does not preserve the exact Unicode anchor surface.")
        if len(continuation) > self.resources.limits.anchor_continuation_tokens:
            raise BudgetExceeded(f"Anchor {anchor.id} exceeds its continuation token budget.")
        if event == "turn_terminated":
            end_token = self.turn_end_token
            if end_token is None:
                raise OmniAnchorError(
                    f"Adapter {self.adapter.name} declares no turn-end token; set "
                    "model.turn_end_token explicitly to score the turn_terminated event."
                )
            if end_token not in getattr(tokenizer, "all_special_tokens", []):
                raise OmniAnchorError(f"Tokenizer does not declare the turn-end token {end_token!r}.")
            end_id = tokenizer.convert_tokens_to_ids(end_token)
            terminated = self._encode(prefix_text + anchor.surface + end_token, frozen)
            end_start = self._check_prefix(full, terminated)
            expected = torch.tensor([[end_id]], dtype=full["input_ids"].dtype)
            if not torch.equal(terminated["input_ids"][:, end_start:].cpu(), expected):
                raise BoundaryError("Turn termination must append exactly the native end token.")
            full = terminated
        # All candidates share the same immutable CPU media tensors. Transfer only
        # the current candidate to CUDA, keeping memory bounded as anchors increase.
        for name in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"):
            if name in prefix:
                full[name] = prefix[name]
        return PreparedCandidate(full, n, anchor, event, len(continuation))

    def _reference(self, item: PreparedCandidate) -> list[float]:
        device_item = PreparedCandidate(
            self._device_inputs(item.inputs), item.prefix_length, item.anchor,
            item.event, item.anchor_token_count,
        )
        return reference_token_logps(self._model, device_item)

    def _single_prefill(self, prefix: dict[str, Any], items: list[PreparedCandidate]) -> dict[str, list[float]]:
        torch = _torch()
        base = getattr(self._model, "model", None)
        if base is not None and hasattr(base, "rope_deltas"):
            base.rope_deltas = None
        with torch.inference_mode():
            out = self._model(**self._device_inputs(prefix), use_cache=False,
                              return_dict=True, logits_to_keep=1)
            distribution = out.logits[0, -1].float().log_softmax(-1)
            candidates = {
                item.anchor.id: [float(distribution[int(item.inputs["input_ids"][0, -1])].item())]
                for item in items
            }
        if self.shared_prefill_validation is None:
            references = {item.anchor.id: self._reference(item) for item in items}
            error = max(abs(candidates[key][0] - references[key][0]) for key in candidates)
            self.shared_prefill_validation = {
                "max_absolute_token_logp_error": error, "tolerance_nats": 0.01,
                "candidate_count": len(items), "passed": math.isfinite(error) and error <= 0.01,
            }
            if not self.shared_prefill_validation["passed"]:
                return references
        if not all(math.isfinite(value[0]) for value in candidates.values()):
            raise OmniAnchorError("Non-finite shared-prefix log probabilities.")
        return candidates

    def score_candidates(
        self, sample: Sample, bridge: Bridge, anchors: Sequence[Anchor],
        *, event: Event = "token_prefix",
    ) -> list[dict[str, Any]]:
        if event not in ("token_prefix", "turn_terminated"):
            raise OmniAnchorError(f"Unsupported score event: {event}")
        if len({anchor.id for anchor in anchors}) != len(anchors):
            raise OmniAnchorError("Duplicate anchor IDs.")
        if not anchors:
            return []
        self._ensure_loaded()
        tokenizer = self._processor.tokenizer
        unsupported = sorted({part.type for part in sample.parts} - self.adapter.modalities)
        if unsupported:
            raise OmniAnchorError(
                f"Adapter {self.adapter.name} accepts {'/'.join(sorted(self.adapter.modalities))} "
                f"only; sample {sample.id!r} contains {', '.join(unsupported)} parts."
                + (" (accepts text only)" if self.adapter.text_only else "")
            )
        text_count = sum(len(tokenizer.encode(part.text or "", add_special_tokens=False))
                         for part in sample.parts if part.type == "text")
        if text_count > self.resources.limits.input_text_tokens:
            raise BudgetExceeded("Input text exceeds its token budget; it was not truncated.")
        frozen = freeze_media(sample, self.resources)
        prefix_text = self.render_prefix(frozen, bridge)
        prefix = self._encode(prefix_text, frozen)
        items = []
        failures: dict[str, dict[str, Any]] = {}
        for anchor in anchors:
            try:
                items.append(self._compile(prefix_text, prefix, frozen, anchor, event))
            except (ResourceUnavailable, _MediaPrefixError):
                raise
            except OmniAnchorError as exc:
                failures[anchor.id] = {
                    "anchor_id": anchor.id, "raw_logp": None, "token_count": None,
                    "event": event, "status": type(exc).__name__, "error": str(exc),
                    "input_fingerprint": frozen.fingerprint,
                }
        self.last_preparation = {
            "input_fingerprint": frozen.fingerprint, "media": frozen.records,
            "input_text_tokens": text_count, "prefix_token_count": prefix["input_ids"].shape[1],
            "rendered_prefix_sha256": hashlib.sha256(prefix_text.encode("utf-8")).hexdigest(),
            "chat_template_sha256": hashlib.sha256(str(getattr(
                self._processor, "chat_template", getattr(tokenizer, "chat_template", ""),
            )).encode("utf-8")).hexdigest(),
            "grids": {name: prefix[name].tolist() for name in ("image_grid_thw", "video_grid_thw")
                      if name in prefix},
        }
        fast: dict[str, list[float]] = {}
        eligible = [item for item in items if event == "token_prefix" and item.anchor_token_count == 1]
        try:
            if self.shared_prefill and eligible and (
                self.shared_prefill_validation is None or self.shared_prefill_validation["passed"]
            ):
                fast = self._single_prefill(prefix, eligible)
            results = []
            for item in items:
                logps = fast.get(item.anchor.id)
                if logps is None:
                    logps = self._reference(item)
                token_ids = item.inputs["input_ids"][0, item.prefix_length:].cpu().tolist()
                results.append({
                    "anchor_id": item.anchor.id, "raw_logp": math.fsum(logps),
                    "token_count": len(logps), "anchor_token_count": item.anchor_token_count,
                    "token_ids": token_ids, "token_logps": logps,
                    "event": event, "status": "ok", "input_fingerprint": frozen.fingerprint,
                })
            by_id = {result["anchor_id"]: result for result in results}
            by_id.update(failures)
            return [by_id[anchor.id] for anchor in anchors]
        except OmniAnchorError:
            raise
        except RuntimeError as exc:
            raise ResourceUnavailable(
                f"Model scoring failed without changing device, precision, or media budget: {exc}"
            ) from exc
