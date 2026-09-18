import pytest

from omnianchor.campaign import create_run, mark_target, merge_score_shards, select_smoke
from omnianchor.types import Part, Sample, TargetSpan


def test_target_rendering_preserves_repeated_word_occurrence_and_unicode():
    sample = Sample(id="target", parts=(Part(type="text", text="河 bank bank"),),
                    target=TargetSpan(part_index=0, start=7, end=11, lemma="bank"))
    result = mark_target(sample)
    assert result.parts[0].text == "河 bank <target>bank</target>"
    assert result.parts[0].text[result.target.start:result.target.end] == "bank"
    assert result.metadata["original_target_text_part"] == sample.parts[0].text
    with pytest.raises(ValueError, match="collides"):
        mark_target(result)


def sample(i, split, group=None):
    return Sample(id=str(i), parts=(Part(type="text", text="material"),),
                  group_id=group or str(i), metadata={"split": split})


def test_smoke_is_order_invariant_and_group_disjoint():
    rows = [sample(i, "train") for i in range(8)]
    rows += [sample(i + 8, "dev", str(i // 2)) for i in range(8)]
    rows += [sample(i + 20, "dev") for i in range(8)]
    rows += [sample(100, "test")]
    selected = select_smoke(rows, 4)
    assert selected == select_smoke(list(reversed(rows)), 4)
    assert len({s.group_id for s in selected}) == 8
    assert {s.metadata["split"] for s in selected} == {"train", "dev"}
    with pytest.raises(ValueError, match="Insufficient"):
        select_smoke([sample(1, "train"), sample(2, "dev", "1")], 1)


def test_run_cannot_overwrite_failure(tmp_path):
    path = tmp_path / "run"
    create_run(path, {"status": "failed"})
    original = (path / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        create_run(path, {"status": "success"})
    assert (path / "manifest.json").read_bytes() == original


def test_merge_rejects_partial_duplicate_or_changed_measurements():
    from omnianchor import Anchor, Bridge, score
    from omnianchor.backends import ToyBackend
    anchors = [Anchor(id="a", surface="care")]
    bridges = [Bridge(id="b", prefix="Associated concept:\n")]
    tables = [score([sample(i, "train")], anchors, bridges, ToyBackend()) for i in range(2)]
    merged = merge_score_shards(tables[::-1], ["0", "1"])
    assert len(merged.frame) == 2
    with pytest.raises(ValueError, match="cover"):
        merge_score_shards(tables[:1], ["0", "1"])
    with pytest.raises(ValueError, match="Duplicate"):
        merge_score_shards([tables[0], tables[0]], ["0", "1"])
    tables[1].manifest["measurement_id"] = "different"
    with pytest.raises(ValueError, match="identities"):
        merge_score_shards(tables, ["0", "1"])


def test_bridge_composition_calibrates_only_identical_instruments():
    from omnianchor import Anchor, Bridge, fit_reference, score, transform
    from omnianchor.backends import ToyBackend
    from omnianchor.campaign import combine_bridge_tables
    anchors = [Anchor(id="a", surface="care")]
    bridges = [Bridge(id=f"b{i}", prefix=f"Relation {i}:\n") for i in range(2)]
    ref = [sample(i, "train") for i in range(3)]
    tables = [score(ref, anchors, [b], ToyBackend()) for b in bridges]
    composed = combine_bridge_tables(tables)
    calibration = fit_reference(composed)
    direct = score([sample(10, "dev")], anchors, bridges, ToyBackend())
    assert transform(direct, calibration).frame.reference_log_ratio.notna().all()
    with pytest.raises(ValueError, match="Duplicate"):
        combine_bridge_tables([tables[0], tables[0]])
    tables[1].manifest["samples"][0]["metadata"]["split"] = "test"
    with pytest.raises(ValueError, match="sample identities"):
        combine_bridge_tables(tables)


def test_coordinate_alias_matches_direct_measurement_without_mutating_source():
    from omnianchor import Anchor, Bridge, score
    from omnianchor.backends import ToyBackend
    from omnianchor.campaign import select_bridge_coordinates
    bridges = [Bridge(id="first", prefix="Associated:\n"), Bridge(id="second", prefix="A concept:\n")]
    anchors = [Anchor(id="a", surface="care")]
    samples = [sample(i, "train") for i in range(2)]
    original = score(samples, anchors, bridges, ToyBackend())
    selected = [bridges[1].model_copy(update={"id": "canonical"})]
    derived = select_bridge_coordinates(original, selected)
    direct = score(samples, anchors, selected, ToyBackend())
    assert derived.manifest["measurement_id"] == direct.manifest["measurement_id"]
    assert derived.frame.raw_logp.tolist() == direct.frame.raw_logp.tolist()
    assert set(original.frame.bridge_id) == {"first", "second"}
    with pytest.raises(ValueError, match="missing"):
        select_bridge_coordinates(original, [Bridge(id="canonical", prefix="Changed wording:\n")])
