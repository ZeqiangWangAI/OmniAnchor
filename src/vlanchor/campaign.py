"""Small campaign helpers: immutable run directories and label-blind smoke selection."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd

from .datasets.common import DatasetBundle
from .io import write_json
from .provenance import stable_hash
from .types import Bridge, Sample, ScoreTable


def mark_target(sample: Sample) -> Sample:
    """Render the exact target occurrence for semantic measurement, preserving raw provenance."""
    if sample.target is None:
        raise ValueError("Target-aware inputs require an explicit span.")
    target = sample.target
    part = sample.parts[target.part_index]
    text = part.text
    if "<target>" in text or "</target>" in text:
        raise ValueError("Source collides with target markers.")
    marked = text[:target.start] + "<target>" + text[target.start:target.end] + "</target>" + text[target.end:]
    parts = list(sample.parts)
    parts[target.part_index] = part.model_copy(update={"text": marked})
    return sample.model_copy(update={"parts": tuple(parts),
        "target": target.model_copy(update={"start": target.start+8, "end": target.end+8}),
        "metadata": {**sample.metadata, "target_rendering": "literal <target> markers v1",
                     "original_target": target.model_dump(), "original_target_text_part": text}})


def create_run(path: Path, manifest: dict) -> Path:
    path.mkdir(parents=True, exist_ok=False)
    write_json(path / "manifest.json", manifest)
    append_event(path, "created")
    return path


def append_event(path: Path, status: str, **details) -> None:
    with (path / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(),
                                 "status": status, **details}, allow_nan=False) + "\n")


def select_smoke(samples: Sequence[Sample], n_per_split: int = 8, seed: int = 42) -> list[Sample]:
    """One material per source group; reference train and dev groups are disjoint.

    Never reads labels or test inputs. Failure does not trigger an alternative seed.
    """
    if n_per_split < 1:
        raise ValueError("Smoke count must be positive.")
    selected, groups = [], set()
    for split in ("train", "dev"):
        candidates = [s for s in samples if s.metadata.get("split") == split]
        candidates.sort(key=lambda s: hashlib.sha256(f"{seed}\0{s.id}".encode()).hexdigest())
        picked = []
        for sample in candidates:
            group = sample.group_id or sample.id
            if group not in groups:
                picked.append(sample)
                groups.add(group)
            if len(picked) == n_per_split:
                break
        if len(picked) != n_per_split:
            raise ValueError(f"Insufficient group-disjoint {split} samples for smoke.")
        selected.extend(picked)
    return selected


def save_bundle(bundle: DatasetBundle, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", bundle.manifest)
    for split in ("train", "dev", "test"):
        subset = [s for s in bundle.samples if s.metadata.get("split") == split]
        folder = output / split
        folder.mkdir()
        write_json(folder / "samples.json", subset)
        bundle.labels.loc[[s.id for s in subset]].to_csv(folder / "labels.csv")


def merge_score_shards(tables: Sequence[ScoreTable], expected_ids: Sequence[str]) -> ScoreTable:
    """Merge sample shards only after identity, exact coverage and duplicate checks."""
    if not tables or not expected_ids or len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Nonempty shards and unique expected sample IDs required.")
    identity = tables[0].manifest["measurement_id"]
    manifest = deepcopy(tables[0].manifest)
    manifest["samples"], manifest["compilation"] = [], {}
    seen = set()
    for table in tables:
        if table.manifest["measurement_id"] != identity:
            raise ValueError("Shard measurement identities differ.")
        ids = {s["id"] for s in table.manifest["samples"]}
        if ids & seen or set(table.frame["sample_id"]) != ids:
            raise ValueError("Duplicate or inconsistent shard sample IDs.")
        seen.update(ids)
        manifest["samples"].extend(table.manifest["samples"])
        manifest["compilation"].update(table.manifest.get("compilation", {}))
    if seen != set(expected_ids):
        raise ValueError("Shards do not cover the frozen sample ID set exactly.")
    frame = pd.concat([t.frame for t in tables], ignore_index=True)
    keys = ["sample_id", "anchor_id", "bridge_id"]
    expected = {(s, a["id"], b["id"]) for s in expected_ids
                for a in manifest["anchors"] for b in manifest["bridges"]}
    if frame.duplicated(keys).any() or set(frame[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError("Shard coordinate grid is incomplete or duplicated.")
    manifest["execution"] = {"shards": len(tables), "score_items": len(frame),
                             "failed_items": int((frame["status"] != "ok").sum()),
                             "elapsed_seconds": sum(t.manifest["execution"]["elapsed_seconds"] for t in tables),
                             "cache_hits": sum(t.manifest["execution"].get("cache_hits", 0) for t in tables)}
    return ScoreTable(frame, manifest)


def combine_bridge_tables(tables: Sequence[ScoreTable]) -> ScoreTable:
    """Compose already measured bridge coordinates; preserve parent IDs and exact instrument."""
    if not tables:
        raise ValueError("Require bridge tables.")
    manifest = deepcopy(tables[0].manifest)
    samples = {s["id"]: s for s in manifest["samples"]}
    bridges, compilation = [], {}
    for table in tables:
        for key in ["instrument", "anchors", "model", "resources", "event"]:
            if table.manifest.get(key) != manifest.get(key):
                raise ValueError(f"Bridge tables differ in {key}.")
        if {s["id"]: s for s in table.manifest["samples"]} != samples:
            raise ValueError("Bridge tables differ in sample identities or metadata.")
        # Reuse the exact-grid validation instead of silently accepting partial candidates.
        merge_score_shards([table], list(samples))
        bridges.extend(table.manifest["bridges"])
        for identifier, records in table.manifest.get("compilation", {}).items():
            compilation.setdefault(identifier, {}).update(records)
    if len({b["id"] for b in bridges}) != len(bridges):
        raise ValueError("Duplicate bridge coordinates.")
    if len({(b["relation"], b["language"]) for b in bridges}) != 1:
        raise ValueError("Bridge relation/language differs.")
    manifest.update(bridges=bridges, compilation=compilation,
        parent_measurement_ids=[t.manifest["measurement_id"] for t in tables])
    manifest["measurement_id"] = stable_hash({k: manifest[k] for k in ["instrument", "anchors", "bridges"]})
    frame = pd.concat([t.frame for t in tables], ignore_index=True)
    frame["measurement_id"] = manifest["measurement_id"]
    manifest["execution"] = {"score_items": len(frame), "failed_items": int((frame.status != "ok").sum()),
        "elapsed_seconds": sum(t.manifest["execution"]["elapsed_seconds"] for t in tables),
        "cache_hits": sum(t.manifest["execution"].get("cache_hits", 0) for t in tables),
        "composition": "no new model calls"}
    return ScoreTable(frame, manifest)


def select_bridge_coordinates(table: ScoreTable, bridges: Sequence[Bridge]) -> ScoreTable:
    """Reuse exact raw coordinates, allowing only bridge-ID aliases; never change wording."""
    if not bridges or len({b.id for b in bridges}) != len(bridges):
        raise ValueError("Require unique requested bridge IDs.")
    if any(c.startswith("reference_") for c in table.frame.columns):
        raise ValueError("Select raw scores before fitting reference calibration.")
    ids = [s["id"] for s in table.manifest["samples"]]
    merge_score_shards([table], ids)
    mapping = {}
    for bridge in bridges:
        desired = {k: v for k, v in bridge.model_dump(mode="json").items() if k != "id"}
        matching = [b["id"] for b in table.manifest["bridges"] if {k: v for k, v in b.items() if k != "id"} == desired]
        if len(matching) != 1 or matching[0] in mapping:
            raise ValueError("Requested bridge is missing, ambiguous, or repeated.")
        mapping[matching[0]] = bridge.id
    manifest = deepcopy(table.manifest)
    manifest["bridges"] = [b.model_dump(mode="json") for b in bridges]
    manifest["compilation"] = {s: {mapping[b]: value for b, value in records.items() if b in mapping}
                               for s, records in manifest.get("compilation", {}).items()}
    manifest["parent_measurement_ids"] = [table.manifest["measurement_id"]]
    manifest["bridge_id_aliases"] = mapping
    manifest["measurement_id"] = stable_hash({k: manifest[k] for k in ["instrument", "anchors", "bridges"]})
    frame = table.frame[table.frame.bridge_id.isin(mapping)].copy()
    frame["bridge_id"] = frame.bridge_id.map(mapping)
    frame["measurement_id"] = manifest["measurement_id"]
    manifest["execution"] = {"score_items": len(frame), "failed_items": int((frame.status != "ok").sum()),
        "elapsed_seconds": 0., "cache_hits": 0, "source_elapsed_seconds": table.manifest["execution"]["elapsed_seconds"],
        "composition": "exact coordinate selection/ID aliases; no neural calls"}
    return ScoreTable(frame, manifest)
