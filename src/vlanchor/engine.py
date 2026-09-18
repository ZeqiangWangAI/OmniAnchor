"""One scoring orchestration path shared by Python, CLI and experiments."""

from __future__ import annotations

import math
from pathlib import Path
from time import perf_counter
from typing import Protocol, Sequence

import pandas as pd

from .errors import MissingMedia, ResourceUnavailable, VLanchorError
from .io import read_json, write_json
from .provenance import file_hash, runtime_manifest, stable_hash
from .types import Anchor, Bridge, Event, ResourceProfile, Sample, ScoreTable, StudySpec


class Backend(Protocol):
    identity: dict

    def score_candidates(self, sample: Sample, bridge: Bridge, anchors: Sequence[Anchor],
                         *, event: Event = "token_prefix") -> list[dict]: ...


def make_backend(spec: StudySpec) -> Backend:
    if spec.model.backend == "toy":
        from .backends import ToyBackend
        return ToyBackend(resources=spec.resources, seed=spec.seed, system_prompt=spec.system_prompt)
    from .backends import HFBackend
    return HFBackend(spec.model, resources=spec.resources, system_prompt=spec.system_prompt)


def sample_fingerprint(sample: Sample) -> str:
    media = []
    for part in sample.parts:
        if part.path:
            path = Path(part.path)
            if not path.is_file():
                raise MissingMedia(f"Missing media: {path}")
            media.append(file_hash(path))
    return stable_hash({"sample": sample.model_dump(mode="json"), "media_sha256": media})


def score(samples: Sequence[Sample], anchors: Sequence[Anchor], bridges: Sequence[Bridge],
          backend: Backend, *, event: Event = "token_prefix", cache: str | Path | None = None,
          include_token_details: bool = False, resources: ResourceProfile | None = None,
          system_prompt: str | None = None) -> ScoreTable:
    if not samples or not anchors or not bridges:
        raise ValueError("Samples, anchors and bridges must all be nonempty.")
    for name, items in (("sample", samples), ("anchor", anchors), ("bridge", bridges)):
        if len({item.id for item in items}) != len(items):
            raise ValueError(f"Duplicate {name} IDs.")
    if len({(b.relation, b.language) for b in bridges}) != 1:
        raise ValueError("Scoring requires a single bridge relation and language.")
    if event not in ("token_prefix", "turn_terminated"):
        raise ValueError("Unsupported scoring event.")
    backend_resources = getattr(backend, "resources", None)
    if resources is not None and backend_resources is not None and resources != backend_resources:
        raise ValueError("Explicit backend resources differ from the declared measurement resources.")
    resources = resources or backend_resources or ResourceProfile()
    if hasattr(backend, "system_prompt"):
        if system_prompt is not None and system_prompt != backend.system_prompt:
            raise ValueError("Explicit backend system_prompt differs from the declared prompt.")
        system_prompt = backend.system_prompt
    instrument = {"model": backend.identity, "event": event,
                  "resources": resources.model_dump(mode="json"), "system_prompt": system_prompt}
    coordinate_config = {"instrument": instrument,
                         "anchors": [a.model_dump(mode="json") for a in anchors],
                         "bridges": [b.model_dump(mode="json") for b in bridges]}
    measurement_id = stable_hash(coordinate_config)
    manifest = {"schema_version": 1, **coordinate_config, "measurement_id": measurement_id,
                "model": backend.identity, "event": event,
                "resources": instrument["resources"], "samples": [], "compilation": {},
                "runtime": runtime_manifest()}
    cache_path = Path(cache) if cache else None
    if cache_path:
        cache_path.mkdir(parents=True, exist_ok=True)
    rows, cache_hits = [], 0
    fatal_resource_error = None
    start = perf_counter()
    for sample in samples:
        sample_error = None
        try:
            content_hash = sample_fingerprint(sample)
        except VLanchorError as exc:
            content_hash, sample_error = None, exc
        manifest["samples"].append({**sample.model_dump(mode="json"), "content_hash": content_hash})
        for bridge in bridges:
            found, pending, keys = {}, [], {}
            for anchor in anchors:
                key = stable_hash({"instrument": instrument, "sample": content_hash,
                                   "bridge": bridge.model_dump(mode="json"),
                                   "anchor": anchor.model_dump(mode="json")})
                keys[anchor.id] = key
                target = cache_path / f"{key}.json" if cache_path else None
                if sample_error is None and target and target.exists():
                    entry = read_json(target)
                    if entry.get("cache_key") != key:
                        raise ValueError("Cache identity mismatch.")
                    found[anchor.id] = entry["score"]
                    cache_hits += 1
                else:
                    pending.append(anchor)
            if pending:
                try:
                    if fatal_resource_error is not None:
                        raise fatal_resource_error
                    if sample_error:
                        raise sample_error
                    output = backend.score_candidates(sample, bridge, pending, event=event)
                    if len(output) != len(pending) or {r["anchor_id"] for r in output} != {a.id for a in pending}:
                        raise ValueError("Backend returned an incomplete or duplicate candidate set.")
                    for item in output:
                        item["_preparation"] = getattr(backend, "last_preparation", {})
                    found.update({r["anchor_id"]: r for r in output})
                except (VLanchorError, FileNotFoundError, FloatingPointError) as exc:
                    if isinstance(exc, ResourceUnavailable):
                        fatal_resource_error = exc
                    found.update({a.id: {"anchor_id": a.id, "status": type(exc).__name__,
                                         "error": str(exc)} for a in pending})
            for anchor in anchors:
                result = dict(found[anchor.id])
                status = result.get("status", "ok")
                if status == "ok":
                    lp, count = result.get("raw_logp"), result.get("token_count")
                    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                        raise ValueError("Backend returned invalid token count.")
                    if lp is None or not math.isfinite(lp) or lp > 1e-8:
                        raise ValueError("Backend returned an invalid log probability.")
                    if result.get("event", event) != event:
                        raise ValueError("Backend event differs from requested event.")
                    result["mean_token_logp"] = lp / count
                    if cache_path and anchor in pending:
                        write_json(cache_path / f"{keys[anchor.id]}.json",
                                   {"cache_key": keys[anchor.id], "score": result})
                else:
                    result.update(raw_logp=None, token_count=None, mean_token_logp=None)
                preparation = result.pop("_preparation", {})
                if preparation:
                    manifest["compilation"].setdefault(sample.id, {})[bridge.id] = preparation
                if not include_token_details:
                    result.pop("token_ids", None)
                    result.pop("token_logps", None)
                rows.append({**result, "sample_id": sample.id, "anchor_id": anchor.id,
                             "anchor_surface": anchor.surface, "bridge_id": bridge.id,
                             "bridge_prefix": bridge.prefix, "relation": bridge.relation,
                             "event": event, "status": status,
                             "model_id": backend.identity.get("id", backend.identity.get("model_id", "unknown")),
                             "model_revision": backend.identity.get("revision", "unknown"),
                             "measurement_id": measurement_id})
    manifest["execution"] = {"elapsed_seconds": perf_counter() - start, "cache_hits": cache_hits,
                             "score_items": len(rows), "failed_items": sum(r["status"] != "ok" for r in rows)}
    manifest["execution"]["shared_prefill_validation"] = getattr(backend, "shared_prefill_validation", None)
    return ScoreTable(pd.DataFrame(rows), manifest)


def measure(samples: Sequence[Sample], spec: StudySpec, *, backend: Backend | None = None,
            cache: str | Path | None = None) -> ScoreTable:
    if backend is not None and hasattr(backend, "system_prompt") and backend.system_prompt != spec.system_prompt:
        raise ValueError("Explicit backend system_prompt differs from StudySpec.")
    return score(samples, spec.anchors, spec.bridges, backend or make_backend(spec),
                 event=spec.event, cache=cache, include_token_details=spec.include_token_details,
                 resources=spec.resources, system_prompt=spec.system_prompt)
