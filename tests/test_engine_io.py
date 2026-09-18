"""Independent engine/cache/storage integration tests with the synthetic byte model."""

import json

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from omnianchor.backends import ToyBackend
from omnianchor.calibration import fit_reference, to_matrix, transform
from omnianchor.engine import measure, sample_fingerprint, score
from omnianchor.errors import IncompatibleMeasurement, MissingScores, ResourceUnavailable
from omnianchor.io import (
    load_calibration, load_matrix, load_samples, load_scores, read_json,
    save_calibration, save_matrix, save_scores, write_json,
)
from omnianchor.types import Anchor, Bridge, Limits, ModelSpec, Part, ResourceProfile, Sample, StudySpec


class CountingToy(ToyBackend):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = []

    def score_candidates(self, sample, bridge, anchors, *, event="token_prefix"):
        self.calls.append((sample.id, bridge.id, tuple(a.id for a in anchors)))
        return super().score_candidates(sample, bridge, anchors, event=event)


class ResourceFailingToy(ToyBackend):
    def __init__(self, successful_calls=0):
        super().__init__()
        self.successful_calls = successful_calls
        self.attempts = 0

    def score_candidates(self, sample, bridge, anchors, *, event="token_prefix"):
        self.attempts += 1
        if self.attempts > self.successful_calls:
            raise ResourceUnavailable("Synthetic CUDA resource failure; do not retry.")
        return super().score_candidates(sample, bridge, anchors, event=event)


def instrument(*, resources=None):
    return StudySpec(
        model=ModelSpec(backend="toy", id="synthetic", revision="fixture", device="cpu", precision="fp32"),
        anchors=(Anchor(id="care", surface="care"), Anchor(id="power", surface="power")),
        bridges=(Bridge(id="b1", prefix="Associated concept:\n"),
                 Bridge(id="b2", prefix="Related concept:\n")),
        resources=resources or ResourceProfile(),
    )


def text_samples(n=4, split="train"):
    return [Sample(id=f"s{i}", parts=(Part(type="text", text=f"Example {i} with varying content."),),
                   group_id=f"source:{i}", metadata={"split": split}) for i in range(n)]


def test_cache_reuses_exact_coordinates_and_can_restore_token_details(tmp_path):
    spec, backend = instrument(), CountingToy()
    first = measure(text_samples(2), spec, backend=backend, cache=tmp_path)
    assert len(backend.calls) == 4
    assert "token_ids" not in first.frame
    replay = measure(text_samples(2), spec.model_copy(update={"include_token_details": True}),
                     backend=backend, cache=tmp_path)
    assert len(backend.calls) == 4
    assert replay.manifest["execution"]["cache_hits"] == 8
    assert all(len(ids) > 0 for ids in replay.frame.token_ids)
    np.testing.assert_allclose(first.frame.raw_logp, replay.frame.raw_logp)
    assert replay.manifest["compilation"] == first.manifest["compilation"]


def test_adding_anchor_does_not_recompute_or_change_existing_coordinates(tmp_path):
    spec, backend = instrument(), CountingToy()
    original = measure(text_samples(1), spec, backend=backend, cache=tmp_path)
    expanded = spec.model_copy(update={"anchors": (*spec.anchors, Anchor(id="new", surface="art"))})
    result = measure(text_samples(1), expanded, backend=backend, cache=tmp_path)
    assert result.manifest["execution"]["cache_hits"] == 4
    assert all(call[2] == ("new",) for call in backend.calls[2:])
    old = result.frame[result.frame.anchor_id != "new"]
    np.testing.assert_allclose(old.raw_logp, original.frame.raw_logp)


def test_media_content_hash_invalidates_cache_even_when_path_and_id_are_unchanged(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (4, 4), "red").save(image_path)
    sample = Sample(id="image", parts=(Part(type="image", path=str(image_path)),))
    spec, backend = instrument(), CountingToy()
    old_hash = sample_fingerprint(sample)
    red = measure([sample], spec, backend=backend, cache=tmp_path / "cache")
    Image.new("RGB", (4, 4), "blue").save(image_path)
    blue = measure([sample], spec, backend=backend, cache=tmp_path / "cache")
    assert sample_fingerprint(sample) != old_hash
    assert blue.manifest["execution"]["cache_hits"] == 0
    assert len(backend.calls) == 4
    assert red.manifest["samples"][0]["content_hash"] != blue.manifest["samples"][0]["content_hash"]
    assert not np.array_equal(red.frame.raw_logp, blue.frame.raw_logp)


def test_missing_media_failure_is_not_cached_and_recovery_is_possible(tmp_path):
    path = tmp_path / "later.png"
    sample = Sample(id="later", parts=(Part(type="image", path=str(path)),))
    spec, backend, cache = instrument(), CountingToy(), tmp_path / "cache"
    missing = measure([sample], spec, backend=backend, cache=cache)
    assert set(missing.frame.status) == {"MissingMedia"}
    assert missing.frame.raw_logp.isna().all()
    assert backend.calls == []
    assert list(cache.glob("*.json")) == []
    Image.new("RGB", (4, 4), "green").save(path)
    recovered = measure([sample], spec, backend=backend, cache=cache)
    assert set(recovered.frame.status) == {"ok"}
    assert len(backend.calls) == 2


def test_text_budget_failure_has_explicit_missing_scores_and_no_cache_entry(tmp_path):
    resources = ResourceProfile(limits=Limits(input_text_tokens=2))
    spec = instrument(resources=resources)
    result = measure(text_samples(1), spec, cache=tmp_path)
    assert set(result.frame.status) == {"BudgetExceeded"}
    assert result.frame.raw_logp.isna().all()
    assert result.frame.token_count.isna().all()
    assert result.manifest["execution"]["failed_items"] == 4
    assert not list(tmp_path.glob("*.json"))
    with pytest.raises(MissingScores):
        to_matrix(result)


def test_fatal_resource_error_is_not_retried_and_returns_the_full_failed_grid(tmp_path):
    backend, spec = ResourceFailingToy(), instrument()
    result = measure(text_samples(3), spec, backend=backend, cache=tmp_path)
    assert backend.attempts == 1
    assert len(result.frame) == 3 * 2 * 2
    assert not result.frame.duplicated(["sample_id", "bridge_id", "anchor_id"]).any()
    assert set(result.frame.status) == {"ResourceUnavailable"}
    assert result.frame.raw_logp.isna().all()
    assert result.frame.token_count.isna().all()
    assert result.manifest["execution"]["failed_items"] == 12
    assert not list(tmp_path.glob("*.json"))


def test_fatal_resource_error_preserves_successful_results_from_earlier_batches(tmp_path):
    backend, spec = ResourceFailingToy(successful_calls=1), instrument()
    result = measure(text_samples(3), spec, backend=backend, cache=tmp_path)
    assert backend.attempts == 2
    assert len(result.frame) == 12
    successful = result.frame[result.frame.status == "ok"]
    assert len(successful) == 2
    assert set(successful.sample_id) == {"s0"}
    assert set(successful.bridge_id) == {"b1"}
    assert np.isfinite(successful.raw_logp).all()
    assert result.manifest["execution"]["failed_items"] == 10
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_fatal_resource_error_still_preserves_cached_cells_before_and_after_failure(tmp_path):
    spec, samples = instrument(), text_samples(3)
    # These cache hits occur on both sides of the first uncached failing batch.
    cached = score([samples[0], samples[2]], spec.anchors, spec.bridges[:1], ToyBackend(), cache=tmp_path)
    backend = ResourceFailingToy()
    result = measure(samples, spec, backend=backend, cache=tmp_path)
    assert backend.attempts == 1
    assert len(result.frame) == 12
    successful = result.frame[result.frame.status == "ok"]
    assert len(successful) == 4
    keys = ["sample_id", "bridge_id", "anchor_id"]
    actual = successful.set_index(keys).raw_logp.sort_index()
    expected = cached.frame.set_index(keys).raw_logp.sort_index()
    pd.testing.assert_series_equal(actual, expected)
    assert result.manifest["execution"]["cache_hits"] == 4
    assert result.manifest["execution"]["failed_items"] == 8
    assert len(list(tmp_path.glob("*.json"))) == 4


def test_explicit_backend_resource_mismatch_cannot_create_false_provenance():
    spec = instrument(resources=ResourceProfile(limits=Limits(input_text_tokens=2)))
    backend = CountingToy()  # This backend would permit a much longer input.
    with pytest.raises(ValueError, match="[Rr]esource"):
        measure(text_samples(1), spec, backend=backend)
    assert backend.calls == []


def test_explicit_backend_system_prompt_mismatch_is_rejected_before_scoring():
    spec = instrument().model_copy(update={"system_prompt": "Use the declared measurement prompt."})
    backend = CountingToy()  # Its actual system prompt is None.
    with pytest.raises(ValueError, match="system_prompt"):
        measure(text_samples(1), spec, backend=backend)
    assert backend.calls == []


def test_bad_anchor_cannot_change_other_anchor_status_between_cold_and_warm_cache(tmp_path):
    resources = ResourceProfile(limits=Limits(anchor_continuation_tokens=2))
    good, bad = Anchor(id="good", surface="a"), Anchor(id="bad", surface="long")
    bridge = Bridge(id="b", prefix="Concept:\n")
    samples = text_samples(1)
    cold = score(samples, [good, bad], [bridge], ToyBackend(resources=resources), cache=tmp_path / "cold")
    score(samples, [good], [bridge], ToyBackend(resources=resources), cache=tmp_path / "warm")
    warm = score(samples, [good, bad], [bridge], ToyBackend(resources=resources), cache=tmp_path / "warm")
    assert cold.frame.set_index("anchor_id").loc["good", "status"] == "ok"
    assert warm.frame.set_index("anchor_id").loc["good", "status"] == "ok"
    assert list(cold.frame.status) == list(warm.frame.status)
    assert cold.frame.set_index("anchor_id").loc["bad", "status"] == "BudgetExceeded"


def test_score_parquet_roundtrip_preserves_split_manifest_and_numeric_scores(tmp_path):
    original = measure(text_samples(3), instrument())
    path = tmp_path / "nested" / "scores.parquet"
    save_scores(original, path)
    restored = load_scores(path)
    pd.testing.assert_frame_equal(original.frame, restored.frame)
    assert restored.manifest == original.manifest
    assert {s["metadata"]["split"] for s in restored.manifest["samples"]} == {"train"}


def test_calibration_json_roundtrip_preserves_sample_std_and_transformation(tmp_path):
    scores = measure(text_samples(4), instrument())
    reference = fit_reference(scores)
    assert reference.statistics.std_logp.gt(0).all()
    path = tmp_path / "calibration.json"
    save_calibration(reference, path)
    restored = load_calibration(path)
    pd.testing.assert_frame_equal(restored.statistics, reference.statistics)
    assert restored.manifest["std_ddof"] == 1
    expected, actual = transform(scores, reference), transform(scores, restored)
    np.testing.assert_allclose(actual.frame.reference_z, expected.frame.reference_z)
    assert np.isfinite(to_matrix(actual, variant="reference_z").values).all()


def test_insufficient_reference_scale_survives_json_as_undefined_not_zero(tmp_path):
    scores = measure(text_samples(1), instrument())
    path = tmp_path / "one-reference.json"
    save_calibration(fit_reference(scores), path)
    payload = read_json(path)
    assert all(row["std_logp"] is None for row in payload["statistics"])
    calibrated = transform(scores, load_calibration(path))
    assert calibrated.frame.reference_z.isna().all()
    assert set(calibrated.frame.reference_z_status) == {"undefined_reference_scale"}
    assert np.isfinite(to_matrix(calibrated, variant="reference_log_ratio").values).all()
    with pytest.raises(MissingScores):
        to_matrix(calibrated, variant="reference_z")


def test_engine_io_does_not_erase_test_split_or_allow_train_override(tmp_path):
    path = tmp_path / "test-scores.parquet"
    save_scores(measure(text_samples(2, split="test"), instrument()), path)
    with pytest.raises(IncompatibleMeasurement, match="never test"):
        fit_reference(load_scores(path), split="train")


@pytest.mark.parametrize("extension", [".npz", ".parquet"])
def test_matrix_roundtrip_preserves_coordinates_and_checks_sidecar_ids(tmp_path, extension):
    matrix = to_matrix(measure(text_samples(3), instrument()))
    path = tmp_path / ("matrix" + extension)
    save_matrix(matrix, path)
    restored = load_matrix(path)
    np.testing.assert_allclose(restored.values, matrix.values)
    assert restored.sample_ids == matrix.sample_ids
    assert restored.anchor_ids == matrix.anchor_ids
    assert restored.manifest == matrix.manifest
    sidecar = path.with_suffix(path.suffix + ".manifest.json")
    payload = read_json(sidecar)
    payload["anchor_ids"] = list(reversed(payload["anchor_ids"]))
    write_json(sidecar, payload)
    with pytest.raises(ValueError, match="coordinate IDs"):
        load_matrix(path)


def test_samples_jsonl_resolves_media_relative_to_its_file_and_preserves_split(tmp_path):
    path = tmp_path / "samples.jsonl"
    path.write_text(json.dumps({"id": "image", "parts": [{"type": "image", "path": "input.png"}],
                                "metadata": {"split": "test"}}) + "\n", encoding="utf-8")
    samples = load_samples(path)
    assert samples[0].parts[0].path == str(tmp_path / "input.png")
    assert samples[0].metadata["split"] == "test"
    path.write_text(path.read_text() * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        load_samples(path)


def test_json_serialization_is_standard_and_leaves_no_temp_files(tmp_path):
    path = tmp_path / "nested" / "data.json"
    write_json(path, {"values": np.array([1, np.nan, np.inf]), "count": np.int64(2)})
    assert read_json(path) == {"values": [1, None, None], "count": 2}
    assert not list(path.parent.glob(".omnianchor-*.tmp"))
