"""Study YAML composition: packs, model aliases and relative paths resolve before validation."""

import pytest
import yaml

from omnianchor.config import MODEL_REGISTRY, model_registry, resolve_study_config
from omnianchor.io import load_spec


def write_yaml(path, value):
    path.write_text(yaml.safe_dump(value, allow_unicode=True), encoding="utf-8")
    return path


BRIDGES = [{"id": "b1", "prefix": "Associated with:\n"}]


def test_inline_configuration_is_unchanged(tmp_path):
    raw = {"name": "inline", "model": {"backend": "toy", "id": "toy", "revision": "v1",
                                       "device": "cpu", "precision": "fp32"},
           "anchors": [{"id": "care", "surface": "care"}], "bridges": BRIDGES}
    assert resolve_study_config(raw, tmp_path) == raw


def test_anchor_and_bridge_packs_resolve_relative_to_the_study_file(tmp_path):
    packs = tmp_path / "packs"
    packs.mkdir()
    write_yaml(packs / "anchors.yaml", {"name": "pack", "anchors": [
        {"id": "joy", "surface": "joy"}, {"id": "fear", "surface": "fear"}]})
    write_yaml(packs / "bridges.yaml", {"bridges": BRIDGES})
    study = write_yaml(tmp_path / "study.yaml", {
        "model": {"backend": "toy", "id": "toy", "revision": "v1", "device": "cpu", "precision": "fp32"},
        "anchors": "packs/anchors.yaml", "bridges": "packs/bridges.yaml"})
    spec = load_spec(study)
    assert [a.id for a in spec.anchors] == ["joy", "fear"]
    assert spec.bridges[0].prefix == "Associated with:\n"


@pytest.mark.parametrize("key", ["anchors", "bridges"])
def test_missing_pack_or_wrong_pack_key_names_the_field(tmp_path, key):
    with pytest.raises(ValueError, match=f"{key}.*not found"):
        resolve_study_config({key: "absent.yaml"}, tmp_path)
    write_yaml(tmp_path / "wrong.yaml", {"other": []})
    with pytest.raises(ValueError, match=f"{key}"):
        resolve_study_config({key: "wrong.yaml"}, tmp_path)


def test_model_alias_expands_to_a_pinned_specification(tmp_path):
    resolved = resolve_study_config({"model": "qwen3.5-4b"}, tmp_path)
    assert resolved["model"]["id"] == "Qwen/Qwen3.5-4B"
    assert resolved["model"]["revision"] == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    assert resolved["model"]["backend"] == "hf"
    with pytest.raises(ValueError, match="Unknown model alias 'nope'"):
        resolve_study_config({"model": "nope"}, tmp_path)


def test_registry_entries_are_complete_and_pinned():
    registry = model_registry()
    assert registry is MODEL_REGISTRY
    assert {"qwen3.5-4b", "qwen3-vl-4b", "qwen3-4b", "smollm3-3b", "toy"} <= set(registry)
    for alias, entry in registry.items():
        assert {"id", "revision", "backend", "adapter"} <= set(entry), alias
        if entry["backend"] == "hf":
            assert len(entry["revision"]) == 40, alias


def test_loaded_alias_study_validates_as_a_study_spec(tmp_path):
    study = write_yaml(tmp_path / "study.yaml", {
        "model": "toy", "anchors": [{"id": "a", "surface": "a"}], "bridges": BRIDGES})
    spec = load_spec(study)
    assert spec.model.backend == "toy"
