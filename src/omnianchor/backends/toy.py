"""Deterministic byte-Markov fixtures. Scores have NO scientific validity."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Sequence

from ..errors import BudgetExceeded, OmniAnchorError
from ..types import Anchor, Bridge, Event, ResourceProfile, Sample
from .media import freeze_media


class ToyBackend:
    """Offline integration fixtures; not an approximation of any language model."""

    def __init__(
        self, resources: ResourceProfile = ResourceProfile(), seed: int = 42,
        system_prompt: str | None = None,
    ):
        self.resources = resources
        self.seed = seed
        self.system_prompt = system_prompt
        self.last_preparation: dict[str, Any] = {}

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "backend": "toy", "id": "omnianchor/toy-byte-markov-v1", "seed": self.seed,
            "scientific_validity": False, "tokenizer": "utf8-bytes-plus-eot-256",
            "system_prompt": self.system_prompt, "resources": self.resources.model_dump(),
        }

    @staticmethod
    def _logp(state: bytes, token: int) -> float:
        logits = [int.from_bytes(hashlib.sha256(state + i.to_bytes(2, "big")).digest()[:4],
                                 "big") / (2**32) * 8 - 4 for i in range(257)]
        maximum = max(logits)
        return logits[token] - maximum - math.log(sum(math.exp(x - maximum) for x in logits))

    def score_candidates(
        self, sample: Sample, bridge: Bridge, anchors: Sequence[Anchor],
        *, event: Event = "token_prefix",
    ) -> list[dict[str, Any]]:
        if event not in ("token_prefix", "turn_terminated"):
            raise OmniAnchorError(f"Unsupported score event: {event}")
        if len({a.id for a in anchors}) != len(anchors):
            raise OmniAnchorError("Duplicate anchor IDs.")
        text_bytes = sum(len((p.text or "").encode("utf-8"))
                         for p in sample.parts if p.type == "text")
        if text_bytes > self.resources.limits.input_text_tokens:
            raise BudgetExceeded("Toy input text exceeds its UTF-8 byte-token budget.")
        if text_bytes + len(bridge.prefix.encode("utf-8")) > (
            self.resources.limits.total_postprocessor_tokens
        ):
            raise BudgetExceeded("Toy prefix exceeds the total byte-token budget.")
        media = freeze_media(sample, self.resources)
        self.last_preparation = {"input_fingerprint": media.fingerprint,
                                 "media": media.records, "synthetic": True}
        prefix = json.dumps([self.seed, self.system_prompt, media.fingerprint, bridge.prefix],
                            ensure_ascii=False).encode("utf-8")
        base_state = hashlib.sha256(prefix).digest()
        results = []
        for anchor in anchors:
            try:
                tokens = self._candidate_tokens(anchor, event, text_bytes, bridge)
            except OmniAnchorError as exc:
                results.append({
                    "anchor_id": anchor.id, "raw_logp": None, "token_count": None,
                    "event": event, "status": type(exc).__name__, "error": str(exc),
                    "synthetic": True, "input_fingerprint": media.fingerprint,
                })
                continue
            state = base_state
            logps = []
            for token in tokens:
                logps.append(self._logp(state, token))
                state = hashlib.sha256(state + token.to_bytes(2, "big")).digest()
            results.append({
                "anchor_id": anchor.id, "raw_logp": math.fsum(logps),
                "token_count": len(tokens), "token_ids": tokens, "token_logps": logps,
                "event": event, "status": "ok", "synthetic": True,
                "input_fingerprint": media.fingerprint,
            })
        return results

    def _candidate_tokens(self, anchor: Anchor, event: Event, text_bytes: int,
                          bridge: Bridge) -> list[int]:
        if any(token in anchor.surface for token in ("<|", "<think>", "</think>")):
            raise OmniAnchorError("Model control tokens are not valid anchor surfaces.")
        try:
            tokens = list(anchor.surface.encode("utf-8", errors="strict"))
        except UnicodeEncodeError as exc:
            raise OmniAnchorError("Anchor contains invalid Unicode surrogate code points.") from exc
        if len(tokens) > self.resources.limits.anchor_continuation_tokens:
            raise BudgetExceeded(f"Anchor {anchor.id} exceeds its byte-token budget.")
        if event == "turn_terminated":
            tokens.append(256)
        if text_bytes + len(bridge.prefix.encode("utf-8")) + len(tokens) > (
            self.resources.limits.total_postprocessor_tokens
        ):
            raise BudgetExceeded("Toy sequence exceeds the total byte-token budget.")
        return tokens
