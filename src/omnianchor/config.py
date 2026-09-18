"""Study YAML composition: anchor/bridge packs and model aliases resolve before validation."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import yaml

_MODEL_SPEC_KEYS = {"backend", "id", "revision", "device", "precision",
                    "attention_implementation", "adapter", "turn_end_token",
                    "chat_template_kwargs"}


def _load_registry() -> dict[str, dict[str, Any]]:
    text = resources.files("omnianchor").joinpath("registry/models.yaml").read_text("utf-8")
    registry = yaml.safe_load(text)
    if not isinstance(registry, dict) or not registry:
        raise ValueError("The packaged model registry is empty or malformed.")
    return registry


MODEL_REGISTRY: dict[str, dict[str, Any]] = _load_registry()


def model_registry() -> dict[str, dict[str, Any]]:
    """Alias -> pinned model entry, as packaged in ``omnianchor/registry/models.yaml``."""
    return MODEL_REGISTRY


def model_spec_from_alias(alias: str) -> dict[str, Any]:
    """Return only the ``ModelSpec`` fields of a registry entry (documentation keys dropped)."""
    try:
        entry = MODEL_REGISTRY[alias]
    except KeyError as exc:
        raise ValueError(f"Unknown model alias {alias!r}; known aliases: "
                         f"{', '.join(sorted(MODEL_REGISTRY))}") from exc
    return {key: value for key, value in entry.items() if key in _MODEL_SPEC_KEYS}


def _load_pack(reference: str, key: str, base_dir: Path) -> list[Any]:
    path = Path(reference)
    if not path.is_absolute():
        path = base_dir / path
    if not path.is_file():
        raise ValueError(f"{key} pack not found: {path}")
    pack = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(pack, dict) or not isinstance(pack.get(key), list) or not pack[key]:
        raise ValueError(f"{key} pack {path} must contain a non-empty top-level {key!r} list.")
    return pack[key]


def resolve_study_config(raw: dict[str, Any], base_dir: str | Path) -> dict[str, Any]:
    """Expand string references in a study mapping; inline mappings pass through unchanged.

    ``model`` may be a registry alias, and ``anchors``/``bridges`` may be paths to pack
    files (relative to ``base_dir``) whose top-level key matches the field name.
    """
    if not isinstance(raw, dict):
        raise ValueError("A study configuration must be a mapping.")
    base_dir = Path(base_dir)
    resolved = dict(raw)
    if isinstance(resolved.get("model"), str):
        resolved["model"] = model_spec_from_alias(resolved["model"])
    for key in ("anchors", "bridges"):
        if isinstance(resolved.get(key), str):
            resolved[key] = _load_pack(resolved[key], key, base_dir)
    return resolved
