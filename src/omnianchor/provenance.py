"""Content-based identities and explicit, minimal runtime provenance."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                         allow_nan=False, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_manifest() -> dict:
    versions = {}
    for package in ("omnianchor", "numpy", "pandas", "scipy", "scikit-learn", "torch",
                    "transformers", "tokenizers", "accelerate", "av", "Pillow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": versions}
