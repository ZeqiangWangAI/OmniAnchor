"""Small synthetic fixtures using the published field/delimiter contracts; no downloads."""

import json

import pandas as pd
import pytest

from vlanchor.datasets import (
    VALUEEVAL_LABELS, general_anchors, grouped_hash_splits, load_dataset, load_dwug,
    load_emobank, load_fmat, load_oasis, load_vatex, load_valueeval, load_viva, load_wic,
    sha256_file,
)
from vlanchor.errors import MissingMedia
from vlanchor.types import Part, Sample


def write_table(tmp_path, name, data, sep=","):
    path = tmp_path / name
    pd.DataFrame(data).to_csv(path, index=False, sep=sep)
    return path


def test_group_splits_stable_to_order_and_group_membership():
    samples = [Sample(id=str(i), parts=(Part(type="text", text="content"),), group_id=str(i // 2))
               for i in range(100)]
    splits = grouped_hash_splits(samples)
    assert splits == grouped_hash_splits(list(reversed(samples)))
    assert all(splits[str(i)] == splits[str(i + 1)] for i in range(0, 100, 2))
    assert set(splits.values()) == {"train", "dev", "test"}
    with pytest.raises(ValueError):
        grouped_hash_splits(samples, fractions=(0.5, 0.5, 0.5))


def test_valueeval_canonical_text_official_split_labels_and_sha(tmp_path):
    arguments = write_table(tmp_path, "arguments-training.tsv", {
        "Argument ID": ["A1", "A2"], "Premise": ["people benefit", "costs increase"],
        "Stance": ["in favor of", "against"], "Conclusion": ["support policy", " Support   Policy "]}, "\t")
    labels = write_table(tmp_path, "labels-training.tsv", {
        "Argument ID": ["A2", "A1"], **{label: [0, 1] for label in VALUEEVAL_LABELS}}, "\t")
    bundle = load_valueeval(arguments, labels, split="train",
                            expected_sha256={arguments.name: sha256_file(arguments)})
    assert bundle.samples[0].parts[0].text == "Premise: people benefit\nStance: in favor of\nConclusion: support policy"
    assert bundle.labels.loc["A1"].sum() == 20
    assert bundle.samples[0].group_id == bundle.samples[1].group_id
    assert bundle.samples[0].metadata["split"] == "train"
    assert bundle.manifest["split_strategy"] == "official_preserved"
    assert len(bundle.manifest["files"][0]["sha256"]) == 64
    with pytest.raises(ValueError, match="SHA256"):
        load_valueeval(arguments, labels, expected_sha256={arguments.name: "0" * 64})


def test_valueeval_rejects_misaligned_ids_and_nonbinary_labels(tmp_path):
    args = write_table(tmp_path, "args.tsv", {"Argument ID": ["A1"], "Premise": ["p"],
                                                "Stance": ["against"], "Conclusion": ["c"]}, "\t")
    gold = {"Argument ID": ["wrong"], **{label: [0] for label in VALUEEVAL_LABELS}}
    labels = write_table(tmp_path, "gold.tsv", gold, "\t")
    with pytest.raises(ValueError, match="different ID"):
        load_valueeval(args, labels)
    gold["Argument ID"] = ["A1"]
    gold[VALUEEVAL_LABELS[0]] = [2]
    labels = write_table(tmp_path, "gold.tsv", gold, "\t")
    with pytest.raises(ValueError, match="binary"):
        load_valueeval(args, labels)


def test_emobank_preserves_official_splits_and_affect_distinction(tmp_path):
    path = write_table(tmp_path, "emobank.csv", {"id": ["x", "y"], "split": ["train", "test"],
        "V": [2, 4], "A": [3, 4], "D": [3, 2], "text": ["same text", "same text"]})
    bundle = load_emobank(path)
    assert bundle.manifest["splits"] == {"x": "train", "y": "test"}
    assert bundle.manifest["groups_crossing_official_splits"]
    assert bundle.samples[1].metadata["split"] == "test"
    assert "not_value" in bundle.manifest["label_kind"]
    assert bundle.manifest["rating_perspective"] == "combined"


def test_emobank_reader_aggregation_uses_author_filter_and_official_join(tmp_path):
    metadata = write_table(tmp_path, "emobank.csv", {"id": ["x", "y"], "split": ["train", "test"],
        "V": [2, 4], "A": [3, 4], "D": [3, 2], "text": ["first", "second"]})
    ratings = write_table(tmp_path, "individual_reader_ratings.csv", {
        "id": ["x", "x", "x", "y"], "V": [1, 3, 5, 4],
        "A": [1, 2, 4, 4], "D": [1, 4, 4, 4]})
    bundle = load_emobank(metadata, rating_perspective="reader", individual_ratings_path=ratings)
    assert list(bundle.labels.loc["x"]) == [4, 3, 4]
    assert bundle.manifest["excluded_without_valid_perspective_ratings"] == ["y"]
    assert bundle.manifest["valid_rating_counts"] == {"x": 2}
    assert bundle.samples[0].metadata["split"] == "train"
    assert bundle.manifest["rating_perspective"] == "reader"
    assert len(bundle.manifest["files"]) == 2
    with pytest.raises(ValueError, match="requires"):
        load_emobank(metadata, rating_perspective="reader")


@pytest.mark.parametrize("text_column", ["Word", "Phrase", "Text"])
def test_chinese_emobank_real_csv_extension_tab_delimiter(tmp_path, text_column):
    path = write_table(tmp_path, "CVAS_all.csv", {text_column: ["和平", "忙碌"],
        "Valence_Mean": [7, 4], "Arousal_Mean": [2, 7], "Valence_SD": [0.5, 1]}, "\t")
    bundle = load_dataset("chinese_emobank", path=path)
    assert list(bundle.labels.columns) == ["V", "A"]
    assert bundle.samples[0].parts[0].text == "和平"
    assert bundle.samples[0].language == "zh"


def test_oasis_metadata_not_in_image_input_and_missing_media(tmp_path):
    path = write_table(tmp_path, "OASIS.csv", {"Item": [1], "Theme": ["Scene 1"],
        "Valence_mean": [4.5], "Arousal_mean": [3.2]})
    with pytest.raises(MissingMedia):
        load_oasis(path, tmp_path)
    bundle = load_oasis(path, tmp_path, require_media=False)
    assert bundle.samples[0].parts[0].type == "image"
    assert bundle.samples[0].parts[0].text is None
    assert bundle.manifest["media"]["missing"] == ["oasis:1"]
    (tmp_path / "Scene 1.jpg").write_bytes(b"fixture only; decoder tested elsewhere")
    assert load_oasis(path, tmp_path).manifest["media"]["available"] == 1


def viva_row():
    return {"index": 1, "image_file": "1.jpg", "answer": "B",
            "action_list": ["A. Leave.", "B. Help."], "reason": "SECRET GOLD REASON",
            "situation_description": "SECRET SCENE DESCRIPTION",
            "values": {"positive": ["Care: SECRET VALUE EXPLANATION"],
                       "negative": ["Ambition: ANOTHER EXPLANATION"]}}


def test_viva_values_condition_on_action_without_reason_leak(tmp_path):
    path = tmp_path / "VIVA_annotation.json"
    path.write_text(json.dumps([viva_row()]))
    bundle = load_viva(path, tmp_path, require_media=False)
    assert bundle.samples[0].parts[1].text == "Proposed action: Help."
    assert {a.surface for a in bundle.candidates["viva:1"]} == {"Care", "Ambition"}
    assert "SECRET" not in json.dumps(bundle.samples[0].model_dump())
    assert bundle.labels["relevant"].sum() == 1
    actions = load_viva(path, tmp_path, task="actions", require_media=False)
    assert len(actions.samples[0].parts) == 1
    assert {a.surface for a in actions.candidates["viva:1"]} == {"Leave.", "Help."}


def test_viva_rejects_path_traversal_and_missing_negative_candidates(tmp_path):
    path = tmp_path / "viva.json"
    row = viva_row()
    row["image_file"] = "../outside.jpg"
    path.write_text(json.dumps([row]))
    with pytest.raises(ValueError, match="escapes"):
        load_viva(path, tmp_path, require_media=False)
    row = viva_row()
    del row["values"]["negative"]
    path.write_text(json.dumps([row]))
    with pytest.raises(ValueError, match="positive and negative"):
        load_viva(path, tmp_path, require_media=False)


def test_vatex_clipped_and_full_video_are_distinguished(tmp_path):
    path = tmp_path / "vatex.json"
    path.write_text(json.dumps([{"videoID": "youtube_id_0010_0020", "enCap": ["A person runs."],
                                 "chCap": ["一个人跑步。"]}]))
    clips = load_vatex(path, tmp_path, require_media=False, split="validation")
    assert clips.samples[0].parts[0].clip_start is None
    assert clips.samples[0].parts[0].path.endswith("youtube_id_0010_0020.mp4")
    assert clips.samples[1].parts[0].text == "A person runs."
    assert clips.samples[0].group_id == clips.samples[1].group_id == "youtube_id"
    assert clips.samples[0].metadata["split"] == "dev"
    full = load_vatex(path, tmp_path, language="zh", videos_are_clips=False, require_media=False)
    assert full.samples[0].parts[0].clip_start == 10
    assert full.samples[0].parts[0].clip_end == 20
    assert full.samples[0].parts[0].path.endswith("youtube_id.mp4")
    assert "跑步" in full.samples[1].parts[0].text


def test_wic_token_offsets_are_converted_to_unicode_character_offsets(tmp_path):
    data = tmp_path / "train.data.txt"
    data.write_text("carry\tV\t2-1\tYou must carry gear .\tSound carries well .\n")
    gold = tmp_path / "train.gold.txt"
    gold.write_text("F\n")
    bundle = load_wic(data, gold, split="train")
    first, second = bundle.samples
    assert first.parts[0].text[first.target.start:first.target.end] == "carry"
    assert second.parts[0].text[second.target.start:second.target.end] == "carries"
    assert bundle.labels.iloc[0]["same_sense"] == 0
    assert first.group_id == second.group_id == "carry"
    assert general_anchors() == general_anchors()
    assert [a.id for a in general_anchors("en")] == [a.id for a in general_anchors("zh")]


def test_dwug_original_tabbed_csv_and_undecidable_judgments(tmp_path):
    uses = write_table(tmp_path, "uses.csv", {"identifier": ["u1", "u2"], "lemma": ["bank", "bank"],
        "context": ["the bank", "a bank"], "grouping": ["old", "new"],
        "indexes_target_token": ["4:8", "2:6"]}, "\t")
    judgments = write_table(tmp_path, "judgments.csv", {"identifier1": ["u1", "u1"],
        "identifier2": ["u2", "u2"], "judgment": [0, 4], "annotator": ["a", "b"]}, "\t")
    bundle = load_dwug(uses, judgments)
    assert bundle.samples[0].time == "old"
    assert list(bundle.labels["is_decidable"]) == [False, True]
    assert bundle.samples[0].parts[0].text[4:8] == "bank"


def test_fmat_is_explicitly_stored_model_data(tmp_path):
    path = write_table(tmp_path, "d1a.csv", {"query": ["A [MASK] works."], "M_word": ["woman"],
        "T_word": ["engineer"], "model": ["bert"], "prob": [0.2]})
    bundle = load_fmat(path)
    assert bundle.labels.iloc[0]["prob"] == 0.2
    assert "not_human" in bundle.manifest["label_kind"]
    assert bundle.manifest["license_api"].endswith("563c1cf88c5e4a3877f9e96a/")


def test_unknown_dataset_fails_explicitly():
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_dataset("not_a_dataset")


def test_dwug_preserves_literal_quotes_and_character_offsets(tmp_path):
    path = tmp_path / "uses.csv"
    path.write_text('identifier\tlemma\tcontext\tgrouping\tindexes_target_token\n'
                    'u1\tbank_nn\t"The bank is open."\t1\t5:9\n'
                    'u2\tbank_nn\t"At the bank, she said, "hello."\t2\t8:12\n')
    bundle = load_dwug(path)
    assert len(bundle.samples) == 2
    for sample in bundle.samples:
        assert sample.parts[0].text.startswith('"')
        assert sample.parts[0].text[sample.target.start:sample.target.end] == "bank"
