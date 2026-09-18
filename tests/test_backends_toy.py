import math
import subprocess
import sys

from PIL import Image
import pytest

from omnianchor.backends import ToyBackend
from omnianchor.engine import score
from omnianchor.errors import OmniAnchorError
from omnianchor.types import Anchor, Bridge, Limits, Part, ResourceProfile, Sample


SAMPLE = Sample(id="s", parts=(Part(type="text", text="A social event."),))
BRIDGE = Bridge(id="b", prefix="This is associated with:\n")


def test_importing_backends_does_not_import_or_load_optional_model_runtime():
    script = (
        "import sys; from omnianchor.backends import HFBackend, ToyBackend; "
        "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules; "
        "assert 'av' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_fixture_is_reproducible_independent_of_candidate_inventory_and_declared_synthetic():
    anchors = [Anchor(id="a", surface="care"), Anchor(id="b", surface="公平", language="zh")]
    b = ToyBackend()
    first = b.score_candidates(SAMPLE, BRIDGE, anchors)
    second = ToyBackend().score_candidates(SAMPLE, BRIDGE, list(reversed(anchors)))
    assert first == list(reversed(second))
    assert b.identity["scientific_validity"] is False
    assert all(r["synthetic"] and math.isfinite(r["raw_logp"]) and r["raw_logp"] < 0 for r in first)
    extended = b.score_candidates(SAMPLE, BRIDGE, [*anchors, Anchor(id="c", surface="order")])
    assert extended[:2] == first
    assert ToyBackend(seed=43).score_candidates(SAMPLE, BRIDGE, anchors) != first


def test_byte_chain_and_eot_conditioning_are_full_sequence_log_probabilities():
    b = ToyBackend()
    anchors = [Anchor(id="a", surface="ab"), Anchor(id="a_prefix", surface="a")]
    rows = b.score_candidates(SAMPLE, BRIDGE, anchors)
    assert rows[0]["token_logps"][0] == rows[1]["raw_logp"]
    assert rows[0]["raw_logp"] == math.fsum(rows[0]["token_logps"])
    terminated = b.score_candidates(SAMPLE, BRIDGE, anchors[:1], event="turn_terminated")[0]
    assert terminated["token_ids"] == [97, 98, 256]
    assert terminated["token_logps"][:2] == rows[0]["token_logps"]
    assert terminated["raw_logp"] < rows[0]["raw_logp"]


def test_native_fixture_pixels_affect_fingerprint_and_scores(tmp_path):
    image = tmp_path / "input.png"
    Image.new("RGB", (8, 4), "red").save(image)
    sample = Sample(id="i", parts=(Part(type="image", path=str(image)),))
    b = ToyBackend()
    anchors = [Anchor(id="a", surface="care")]
    red = b.score_candidates(sample, BRIDGE, anchors)
    Image.new("RGB", (8, 4), "blue").save(image)
    blue = b.score_candidates(sample, BRIDGE, anchors)
    assert red[0]["raw_logp"] != blue[0]["raw_logp"]
    assert red[0]["input_fingerprint"] != blue[0]["input_fingerprint"]
    assert b.last_preparation["media"][0]["width"] == 8


def test_fixture_rejects_control_tokens_duplicate_ids_and_budget_overruns():
    b = ToyBackend()
    rejected = b.score_candidates(SAMPLE, BRIDGE, [Anchor(id="a", surface="<|im_end|>")])[0]
    assert rejected["status"] == "OmniAnchorError" and "control tokens" in rejected["error"]
    with pytest.raises(OmniAnchorError, match="Duplicate"):
        b.score_candidates(SAMPLE, BRIDGE, [Anchor(id="a", surface="a"), Anchor(id="a", surface="b")])
    b = ToyBackend(resources=ResourceProfile(limits=Limits(anchor_continuation_tokens=1)))
    assert b.score_candidates(SAMPLE, BRIDGE, [Anchor(id="a", surface="公平")])[0]["status"] == "BudgetExceeded"


def test_bad_anchor_append_keeps_valid_scores_and_cache_status_independent(tmp_path):
    b = ToyBackend(resources=ResourceProfile(limits=Limits(anchor_continuation_tokens=1)))
    good = Anchor(id="good", surface="a")
    bad = Anchor(id="bad", surface="too long")
    standalone = b.score_candidates(SAMPLE, BRIDGE, [good])[0]
    together = b.score_candidates(SAMPLE, BRIDGE, [good, bad])
    assert together[0] == standalone
    assert together[1]["status"] == "BudgetExceeded" and together[1]["raw_logp"] is None
    cold = score([SAMPLE], [good, bad], [BRIDGE], b)
    score([SAMPLE], [good], [BRIDGE], b, cache=tmp_path)
    warm = score([SAMPLE], [good, bad], [BRIDGE], b, cache=tmp_path)
    assert cold.frame.status.tolist() == warm.frame.status.tolist()
    assert cold.frame.iloc[0].raw_logp == warm.frame.iloc[0].raw_logp
    assert warm.manifest["execution"]["cache_hits"] == 1
