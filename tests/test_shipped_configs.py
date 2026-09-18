"""Every configuration shipped in the repository loads without touching model weights."""

from pathlib import Path

import pytest
import yaml

from omnianchor.config import MODEL_REGISTRY
from omnianchor.backends.adapters import ADAPTERS
from omnianchor.io import load_spec
from omnianchor.pipeline import load_pipeline

ROOT = Path(__file__).resolve().parents[1]
STUDIES = sorted((ROOT / "configs/studies").glob("*.yaml")) + sorted((ROOT / "configs/smoke").glob("*.json"))


@pytest.mark.parametrize("path", STUDIES, ids=lambda p: p.name)
def test_study_configurations_validate(path):
    spec = load_spec(path)
    assert spec.anchors and spec.bridges
    if spec.model.backend == "hf":
        assert len(spec.model.revision) == 40


@pytest.mark.parametrize("path", sorted((ROOT / "configs/bridges").glob("*.yaml")), ids=lambda p: p.name)
def test_bridge_packs_use_one_relation_and_language(path):
    pack = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert len({(b["relation"], b["language"]) for b in pack["bridges"]}) == 1


def test_pipeline_example_and_smoke_model_list():
    spec, base = load_pipeline(ROOT / "configs/pipelines/toy_quickstart.yaml")
    assert (base / spec.samples).is_file()
    aliases = yaml.safe_load((ROOT / "configs/smoke/models.yaml").read_text())["models"]
    for alias in aliases:
        entry = MODEL_REGISTRY[alias]
        assert entry["adapter"] in ADAPTERS, alias


def test_registry_revisions_agree_with_the_historical_pin_record():
    import json
    pins = json.loads((ROOT / "configs/models-20260910.json").read_text())["models"]
    for entry in MODEL_REGISTRY.values():
        if entry["id"] in pins:
            assert entry["revision"] == pins[entry["id"]], entry["id"]
