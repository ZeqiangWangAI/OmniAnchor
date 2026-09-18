"""GPU acceptance checks against an already-loaded backend; never loads weights.

Import ``verify_native`` from the demo process. Full logits are materialized only
for a bounded text sequence, while both scoring paths normalize in FP32 without
changing model precision. Existing demo evidence can avoid repeated checks.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

from omnianchor.backends import HFBackend
from omnianchor.backends.hf import PreparedCandidate
from omnianchor.backends.media import freeze_media
from omnianchor.errors import BudgetExceeded, ResourceUnavailable, OmniAnchorError
from omnianchor.io import write_json
from omnianchor.types import Anchor, Bridge, Part, Sample


def _full_logps(backend: HFBackend, item: PreparedCandidate) -> list[float]:
    import torch

    inputs = backend._device_inputs(item.inputs)
    base = getattr(backend._model, "model", None)
    if base is not None and hasattr(base, "rope_deltas"):
        base.rope_deltas = None
    with torch.inference_mode():
        output = backend._model(**inputs, use_cache=False, return_dict=True, logits_to_keep=0)
        ids = inputs["input_ids"]
        if output.logits.shape[:2] != ids.shape:
            raise OmniAnchorError("logits_to_keep=0 did not return all sequence positions.")
        p = item.prefix_length
        # Independent full-output reference: position p-1 predicts target token p.
        target_logits = output.logits[0, p - 1:-1, :].float()
        target_ids = ids[0, p:]
        values = target_logits.log_softmax(-1).gather(1, target_ids[:, None]).squeeze(1)
        if not bool(torch.isfinite(values).all()):
            raise OmniAnchorError("Non-finite full-logit reference probabilities.")
        return values.cpu().tolist()


def _comparison(left: list[float], right: list[float], tolerance: float) -> dict[str, Any]:
    if len(left) != len(right) or not left:
        raise OmniAnchorError("Native verification requires matching nonempty token chains.")
    errors = [abs(a - b) for a, b in zip(left, right)]
    maximum = max(errors)
    return {"passed": math.isfinite(maximum) and maximum <= tolerance,
            "max_absolute_token_logp_error": maximum, "token_errors": errors,
            "tolerance_nats_per_token": tolerance}


def _successful(rows: list[dict]) -> None:
    if any(row.get("status") != "ok" for row in rows):
        raise OmniAnchorError(f"Native verification candidate failed: {rows}")


def _shared_evidence(backend: HFBackend, evidence: dict, source: str,
                     comparisons: list[dict] | None = None) -> dict[str, Any]:
    validation = dict(evidence.get("runtime_validation") or evidence)
    if not isinstance(validation.get("passed"), bool):
        raise OmniAnchorError("Shared prefill evidence needs an explicit passed boolean.")
    accepted = validation["passed"] and all(row["passed"] for row in comparisons or [])
    if not accepted:
        # Numerical disagreement disables this optional optimization. The strict
        # uncached reference remains a valid production path, not a failed demo.
        validation["passed"] = False
        backend.shared_prefill_validation = validation
    return {"passed": True, "source": source,
            "optimization_status": "accepted" if accepted else "disabled",
            "fallback": None if accepted else "uncached_reference",
            "runtime_validation": validation, "candidates": comparisons or []}


def verify_native(
    backend: HFBackend, *, output: str | Path | None = None,
    sample: Sample | None = None, bridge: Bridge | None = None,
    anchors: Sequence[Anchor] | None = None,
    image_sample: Sample | None = None, video_sample: Sample | None = None,
    existing_checks: dict[str, dict[str, Any]] | None = None,
    tolerance_nats: float = 0.01, max_full_sequence_tokens: int = 256,
) -> dict[str, Any]:
    """Compare native selected/full logits, optionally checking media-state reuse.

    ``existing_checks`` may supply ``shared_prefill`` and/or ``modality_state``
    dictionaries with explicit ``passed`` booleans from the same live demo. They
    are preserved as reused evidence, not represented as newly executed checks.
    A failed optional prefill comparison safely disables that optimization;
    selected/full-logit or modality-state disagreement fails the acceptance.
    A missing model/processor fails before any download or device fallback.
    """
    if backend._model is None or backend._processor is None:
        raise ResourceUnavailable("Native verification requires an already-loaded backend.")
    if getattr(backend._model, "training", False):
        raise OmniAnchorError("Native verification requires model.eval().")
    if not math.isfinite(tolerance_nats) or tolerance_nats <= 0:
        raise OmniAnchorError("Native verification tolerance must be finite and positive.")
    sample = sample or Sample(id="native-check", parts=(Part(type="text", text="A red circle."),))
    bridge = bridge or Bridge(id="native-check", prefix="An associated concept is:\n")
    anchors = tuple(anchors) if anchors is not None else (
        Anchor(id="red", surface="red"), Anchor(id="circle", surface="blue circle"),
    )
    if not anchors or len(anchors) > 4 or len({a.id for a in anchors}) != len(anchors):
        raise OmniAnchorError("Native full-logit verification needs one to four unique anchors.")
    if any(part.type != "text" for part in sample.parts):
        raise OmniAnchorError("Full-logit verification uses a bounded text-only input.")
    frozen = freeze_media(sample, backend.resources)
    messages = []
    if backend.system_prompt is not None:
        messages.append({"role": "system", "content": backend.system_prompt})
    messages.append({"role": "user", "content": frozen.content})
    prefix_text = backend._processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    ) + bridge.prefix
    prefix = backend._encode(prefix_text, frozen)
    items = [backend._compile(prefix_text, prefix, frozen, a, "token_prefix") for a in anchors]
    if any(item.inputs["input_ids"].shape[1] > max_full_sequence_tokens for item in items):
        raise BudgetExceeded("Verification sequence exceeds its explicit full-logit memory bound.")

    comparisons = []
    for item in items:
        selected, full = backend._reference(item), _full_logps(backend, item)
        comparisons.append({
            "anchor_id": item.anchor.id, "token_count": len(full),
            "token_ids": item.inputs["input_ids"][0, item.prefix_length:].tolist(),
            "selected_token_logps": selected, "full_token_logps": full,
            **_comparison(selected, full, tolerance_nats),
        })
    checks: dict[str, dict[str, Any]] = {"selected_vs_full": {
        "passed": all(row["passed"] for row in comparisons), "source": "executed",
        "normalization_dtype": "float32", "event": "token_prefix",
        "full_logit_sequence_limit": max_full_sequence_tokens,
        "sequence_lengths": [item.inputs["input_ids"].shape[1] for item in items],
        "candidates": comparisons,
    }}
    existing_checks = existing_checks or {}
    for key in ("shared_prefill", "modality_state"):
        if key in existing_checks:
            if not isinstance(existing_checks[key].get("passed"), bool):
                raise OmniAnchorError(f"Reused {key} evidence needs an explicit passed boolean.")
            checks[key] = (
                _shared_evidence(backend, existing_checks[key], "reused_live_demo_evidence")
                if key == "shared_prefill" else
                {**existing_checks[key], "source": "reused_live_demo_evidence"}
            )

    if "shared_prefill" not in checks:
        single = [item for item in items if item.anchor_token_count == 1]
        if single:
            fast = backend._single_prefill(prefix, single)
            reference_by_id = {row["anchor_id"]: row["selected_token_logps"] for row in comparisons}
            compared = [{"anchor_id": item.anchor.id,
                         **_comparison(fast[item.anchor.id], reference_by_id[item.anchor.id], tolerance_nats)}
                        for item in single]
            validated = backend.shared_prefill_validation
            checks["shared_prefill"] = _shared_evidence(backend, validated, "executed", compared)
        else:
            checks["shared_prefill"] = {"source": "not_run", "reason": "No single-token anchor."}

    if "modality_state" not in checks:
        media_samples = [("image", image_sample), ("video", video_sample)]
        media_samples = [(kind, media) for kind, media in media_samples if media is not None]
        comparisons_by_modality = []
        previous_fast = backend.shared_prefill
        try:
            backend.shared_prefill = False
            baseline = backend.score_candidates(sample, bridge, anchors) if media_samples else []
            _successful(baseline)
            for kind, media in media_samples:
                if not any(part.type == kind for part in media.parts):
                    raise OmniAnchorError(f"The {kind} state check needs an actual {kind} input part.")
                _successful(backend.score_candidates(media, bridge, anchors[:1]))
                preparation = backend.last_preparation
                grid = preparation.get("grids", {}).get(f"{kind}_grid_thw")
                if not grid:
                    raise OmniAnchorError(f"The {kind} forward did not record its native grid.")
                replay = backend.score_candidates(sample, bridge, anchors)
                _successful(replay)
                token_checks = [
                    {"anchor_id": before["anchor_id"],
                     **_comparison(before["token_logps"], after["token_logps"], tolerance_nats)}
                    for before, after in zip(baseline, replay)
                ]
                comparisons_by_modality.append({
                    "modality": kind, "grid": grid, "media": preparation.get("media"),
                    "passed": all(check["passed"] for check in token_checks), "candidates": token_checks,
                })
        finally:
            backend.shared_prefill = previous_fast
        checks["modality_state"] = (
            {"passed": all(row["passed"] for row in comparisons_by_modality),
             "source": "executed", "order": ["text", *[part for kind, _ in media_samples
                                                       for part in (kind, "text")]],
             "comparisons": comparisons_by_modality}
            if media_samples else {"source": "not_run", "reason": "No media samples supplied."}
        )
    report = {"status": "passed" if all(c.get("passed", True) for c in checks.values()) else "failed",
              "purpose": "native_implementation_acceptance_not_construct_validation",
              "model": backend.identity, "checks": checks}
    if output is not None:
        write_json(output, report)
    return report


if __name__ == "__main__":
    raise SystemExit("Import verify_native(backend, ...) inside the process with the loaded model.")
