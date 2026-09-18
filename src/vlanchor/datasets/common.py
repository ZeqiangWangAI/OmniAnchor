"""Local data contracts, deterministic splits, and file provenance. Never downloads data."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from ..types import Anchor, Sample


@dataclass
class DatasetBundle:
    samples: list[Sample]
    labels: pd.DataFrame
    manifest: dict[str, Any]
    candidates: dict[str, tuple[Anchor, ...]] = field(default_factory=dict)

    def __post_init__(self):
        splits = self.manifest.get("splits", {})
        for sample in self.samples:
            previous = sample.metadata.get("split")
            if previous is not None and sample.id in splits and previous != splits[sample.id]:
                raise ValueError("Sample split metadata conflicts with dataset manifest.")
        self.samples = [sample.model_copy(update={"metadata": {**sample.metadata, "split": splits[sample.id]}})
                        if sample.id in splits else sample for sample in self.samples]
        ids = [sample.id for sample in self.samples]
        if len(ids) != len(set(ids)):
            raise ValueError("Dataset contains duplicate sample IDs.")
        if set(self.candidates) - set(ids):
            raise ValueError("Candidate sample IDs are absent from samples.")
        for anchors in self.candidates.values():
            if len({a.id for a in anchors}) != len(anchors):
                raise ValueError("Per-sample anchor IDs must be unique.")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_group(text: str) -> str:
    return hashlib.sha256(" ".join(text.casefold().split()).encode("utf-8")).hexdigest()


def grouped_hash_splits(
    samples: Sequence[Sample], seed: int = 42,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> dict[str, str]:
    """Group-stable expected proportions, not exact sample counts. Order invariant."""
    if len(fractions) != 3 or any(f < 0 for f in fractions) or abs(sum(fractions) - 1) > 1e-9:
        raise ValueError("Split fractions must be three nonnegative numbers summing to one.")
    result = {}
    for sample in samples:
        group = sample.group_id or sample.id
        digest = hashlib.sha256(f"{seed}\0{group}".encode("utf-8")).digest()
        u = int.from_bytes(digest[:8], "big") / 2**64
        result[sample.id] = ("train" if u < fractions[0] else
                             "dev" if u < fractions[0] + fractions[1] else "test")
    return result


def build_manifest(
    name: str, paths: Sequence[str | Path], samples: Sequence[Sample], *,
    source_url: str, license_note: str, seed: int = 42,
    official_splits: Mapping[str, str] | None = None,
    expected_sha256: Mapping[str, str] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    files = []
    for value in paths:
        path = Path(value).expanduser().resolve(strict=True)
        digest = sha256_file(path)
        expected = (expected_sha256 or {}).get(str(path), (expected_sha256 or {}).get(path.name))
        if expected is not None and digest.lower() != expected.lower():
            raise ValueError(f"SHA256 mismatch for {path.name}")
        files.append({"path": str(path), "sha256": digest, "size_bytes": path.stat().st_size})
    if expected_sha256:
        observed = {f["path"] for f in files} | {Path(f["path"]).name for f in files}
        if set(expected_sha256) - observed:
            raise ValueError("Expected SHA256 entries include files not consumed by this adapter.")
    if official_splits is not None:
        if set(official_splits) != {s.id for s in samples}:
            raise ValueError("Official split mapping must cover every sample exactly.")
        aliases = {"training": "train", "validation": "dev", "valid": "dev", "development": "dev"}
        splits = {k: aliases.get(str(v).lower(), str(v).lower()) for k, v in official_splits.items()}
        if set(splits.values()) - {"train", "dev", "test"}:
            raise ValueError("Unrecognized official split.")
        strategy = "official_preserved"
    else:
        splits, strategy = grouped_hash_splits(samples, seed), "grouped_sha256_60_20_20"
    group_splits: dict[str, set[str]] = {}
    for s in samples:
        group_splits.setdefault(s.group_id or s.id, set()).add(splits[s.id])
    return {
        "schema_version": 1, "dataset": name, "source_url": source_url,
        "license_note": license_note, "files": files, "sample_count": len(samples),
        "seed": seed, "split_strategy": strategy, "splits": splits,
        "groups_crossing_official_splits": sorted(k for k, v in group_splits.items() if len(v) > 1),
        **extra,
    }


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, keep_default_na=False)
    with path.open(encoding="utf-8-sig") as handle:
        header = handle.readline()
    # Chinese EmoBank releases use a .csv extension for tab-delimited tables.
    separator = "\t" if "\t" in header else ","
    return pd.read_csv(path, sep=separator, keep_default_na=False, encoding="utf-8-sig")


def require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}; found {list(frame.columns)}")


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def local_media(root: str | Path, filename: str) -> Path:
    """Reject absolute/traversal paths in downloaded annotation files."""
    root = Path(root).expanduser().resolve()
    relative = Path(filename)
    if relative.is_absolute():
        raise ValueError("Media filenames in annotations must be relative to media_root.")
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("Annotation media path escapes media_root.")
    return candidate
