import json
from types import SimpleNamespace

import pytest

from vlanchor.backends import HFBackend
from vlanchor.backends import hf
from vlanchor.types import ModelSpec


COMMIT = "4815a0a6a064214f2d8208c094464a5a6b76ca8d"


@pytest.mark.parametrize("direct, expected", [
    ({"url": f"https://github.com/huggingface/transformers/archive/{COMMIT}.tar.gz"}, COMMIT),
    ({"url": f"https://codeload.github.com/huggingface/transformers/zip/{COMMIT}"}, COMMIT),
    ({"url": "https://github.com/huggingface/transformers.git",
      "vcs_info": {"vcs": "git", "commit_id": COMMIT.upper()}}, COMMIT),
    ({"url": f"https://private:do-not-output@github.com/huggingface/transformers/archive/{COMMIT}.zip?token=secret"}, COMMIT),
    ({"url": f"https://unrelated.example/transformers/archive/{COMMIT}.tar.gz"}, None),
    ({"url": "https://github.com/huggingface/transformers/archive/main.tar.gz"}, None),
    ({"vcs_info": {"commit_id": "not-an-immutable-commit"}}, None),
    ({"url": "https://[invalid"}, None),
    ([], None),
])
def test_source_revision_parses_immutable_hash_without_exposing_url(monkeypatch, direct, expected):
    distribution = SimpleNamespace(read_text=lambda name: json.dumps(direct))
    monkeypatch.setattr(hf.metadata, "distribution", lambda package: distribution)
    assert hf._transformers_source_revision() == expected


@pytest.mark.parametrize("raw", [None, "not-json", '{"url": 4}'])
def test_missing_or_invalid_direct_url_is_unavailable(monkeypatch, raw):
    monkeypatch.setattr(hf.metadata, "distribution", lambda package: SimpleNamespace(read_text=lambda _: raw))
    assert hf._transformers_source_revision() is None


def test_identity_handles_missing_transformers_and_injected_runtime(monkeypatch):
    versions = {"torch": "fixture-torch", "tokenizers": "fixture-tokenizers", "Pillow": "fixture-pillow",
                "causal-conv1d": "fixture-convolution", "flash-linear-attention": "fixture-linear"}

    def version(package):
        if package not in versions:
            raise hf.metadata.PackageNotFoundError(package)
        return versions[package]

    def missing_distribution(package):
        raise hf.metadata.PackageNotFoundError(package)

    monkeypatch.setattr(hf.metadata, "version", version)
    monkeypatch.setattr(hf.metadata, "distribution", missing_distribution)
    model = SimpleNamespace(eval=lambda: None)
    spec = ModelSpec(device="cpu")
    identity = HFBackend(spec, model=model, processor=object()).identity
    assert identity["injected_test_runtime"] is True
    assert identity["processor_revision"] == spec.revision
    assert identity["transformers_version"] is None
    assert identity["transformers_source_revision"] is None
    assert identity["tokenizers_version"] == "fixture-tokenizers"
    assert identity["pillow_version"] == "fixture-pillow"
    assert identity["causal_conv1d_version"] == "fixture-convolution"
    assert identity["flash_linear_attention_version"] == "fixture-linear"
    versions.clear()
    absent = HFBackend(spec).identity
    assert absent["tokenizers_version"] is absent["pillow_version"] is None
    assert absent["causal_conv1d_version"] is absent["flash_linear_attention_version"] is None
