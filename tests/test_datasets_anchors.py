"""Train-only anchor construction from synthetic records in the published WiC schema."""

import pytest

from omnianchor.datasets import DatasetBundle, build_general_anchors, load_wic


def wic_bundle(tmp_path, lemmas, split="train"):
    path = tmp_path / "data.txt"
    path.write_text("".join(f"{lemma}\tN\t0-0\t{lemma} context one\t{lemma} context two\n"
                            for lemma in lemmas), encoding="utf-8")
    return load_wic(path, split=split)


def test_general_anchors_count_each_pair_once_and_sort_frequency_then_surface(tmp_path):
    bundle = wic_bundle(tmp_path, ["zebra", "apple", "zebra", "berry"])
    # Repeated usage records must not inflate the single apple pair's frequency.
    apple = bundle.samples[2]
    extra = [apple.model_copy(update={"id": f"duplicate:{i}"}) for i in range(6)]
    manifest = {**bundle.manifest, "splits": {**bundle.manifest["splits"],
                                            **{sample.id: "train" for sample in extra}}}
    repeated = DatasetBundle(bundle.samples + extra, bundle.labels, manifest)
    anchors = build_general_anchors(repeated, n=3)
    assert [anchor.surface for anchor in anchors] == ["zebra", "apple", "berry"]
    assert anchors == build_general_anchors(bundle, n=3)
    reordered = DatasetBundle(list(reversed(repeated.samples)), repeated.labels, manifest)
    assert anchors == build_general_anchors(reordered, n=3)


def test_general_anchor_64_128_256_packs_have_identical_prefix_ids(tmp_path):
    bundle = wic_bundle(tmp_path, [f"word{i:03d}" for i in reversed(range(256))])
    full = build_general_anchors(bundle, n=256)
    assert full[:64] == build_general_anchors(bundle, n=64)
    assert full[:128] == build_general_anchors(bundle)
    assert [anchor.surface for anchor in full] == [f"word{i:03d}" for i in range(256)]
    assert len({anchor.id for anchor in full}) == 256
    assert all(anchor.id == anchor.concept_id and anchor.language == "en" for anchor in full)


@pytest.mark.parametrize("split", ["dev", "test", None])
def test_general_anchors_reject_nontrain_or_undeclared_source(tmp_path, split):
    bundle = wic_bundle(tmp_path, ["bank"], split=split)
    with pytest.raises(ValueError, match="explicitly declared WiC train"):
        build_general_anchors(bundle, n=1)


def test_general_anchors_reject_nonwic_and_insufficient_unique_lemmas(tmp_path):
    bundle = wic_bundle(tmp_path, ["bank", "bank"])
    with pytest.raises(ValueError, match="only 1 unique lemmas"):
        build_general_anchors(bundle, n=2)
    other = DatasetBundle(bundle.samples, bundle.labels, {**bundle.manifest, "dataset": "dwug"})
    with pytest.raises(ValueError, match="WiC dataset"):
        build_general_anchors(other, n=1)


@pytest.mark.parametrize("n", [0, -1, 1.5, True])
def test_general_anchors_reject_invalid_size(tmp_path, n):
    with pytest.raises(ValueError, match="positive integer"):
        build_general_anchors(wic_bundle(tmp_path, ["bank"]), n=n)


def test_general_anchors_reject_inconsistent_pair_lemmas(tmp_path):
    bundle = wic_bundle(tmp_path, ["bank"])
    sample = bundle.samples[1]
    changed = sample.model_copy(update={"target": sample.target.model_copy(update={"lemma": "river"})})
    bundle = DatasetBundle([bundle.samples[0], changed], bundle.labels, bundle.manifest)
    with pytest.raises(ValueError, match="inconsistent target lemmas"):
        build_general_anchors(bundle, n=1)
