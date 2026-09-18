"""Published multimodal annotations backed exclusively by user-supplied local media."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from ..errors import MissingMedia
from ..types import Anchor, Part, Sample
from .common import DatasetBundle, build_manifest, local_media, read_json, read_table, require_columns


def _availability(paths: dict[str, Path], required: bool) -> dict:
    missing = [identifier for identifier, path in paths.items() if not path.is_file()]
    if required and missing:
        raise MissingMedia(f"Missing local media for {len(missing)} samples; first IDs: {missing[:5]}")
    return {"available": len(paths) - len(missing), "missing": missing,
            "media_sha256": "not_computed", "metadata_only_allowed": not required}


def load_oasis(
    metadata_path: str | Path, media_root: str | Path, *,
    filename_template: str = "{Theme}.jpg", filename_column: str | None = None,
    require_media: bool = True, seed: int = 42,
    expected_sha256: Mapping[str, str] | None = None,
) -> DatasetBundle:
    """Read OASIS.csv. Override filename_template for a locally renamed image tree."""
    data = read_table(metadata_path)
    require_columns(data, ["Theme", "Valence_mean", "Arousal_mean"], "OASIS")
    if filename_column is not None:
        require_columns(data, [filename_column], "OASIS filenames")
    samples, files, records = [], {}, []
    for i, row in enumerate(data.to_dict("records")):
        identifier = f"oasis:{row.get('Item', i)}"
        filename = str(row[filename_column]) if filename_column else filename_template.format_map(row)
        path = local_media(media_root, filename)
        files[identifier] = path
        samples.append(Sample(id=identifier, source="oasis", group_id=str(row["Theme"]),
                              parts=(Part(type="image", path=str(path)),)))
        records.append({"sample_id": identifier, "V": float(row["Valence_mean"]),
                        "A": float(row["Arousal_mean"])})
    labels = pd.DataFrame(records, columns=["sample_id", "V", "A"]).set_index("sample_id")
    if not np.isfinite(labels.to_numpy()).all():
        raise ValueError("OASIS ratings must be finite.")
    manifest = build_manifest(
        "oasis", [metadata_path], samples, seed=seed, expected_sha256=expected_sha256,
        source_url="https://osf.io/6pnd7/", license_note="See original OASIS source terms.",
        label_kind="human_affect_VA_not_value", media=_availability(files, require_media),
        filename_template=filename_template, filename_column=filename_column,
    )
    return DatasetBundle(samples, labels, manifest)


def load_viva(
    annotation_path: str | Path, media_root: str | Path, *,
    task: str = "values", require_media: bool = True, seed: int = 42,
    expected_sha256: Mapping[str, str] | None = None,
) -> DatasetBundle:
    """VIVA_annotation.json. Values are conditioned on the annotated correct action.

    Reasons, situation descriptions, and explanatory value suffixes are never fed
    to the model. In actions mode, only the image is an input and actions are candidates.
    """
    if task not in {"values", "actions"}:
        raise ValueError("VIVA task must be values or actions.")
    data = read_json(annotation_path)
    if not isinstance(data, list):
        raise ValueError("VIVA annotation must be a JSON list.")
    samples, records, candidates, files = [], [], {}, {}
    for row in data:
        required = {"index", "image_file", "action_list", "answer"}
        if required - row.keys():
            raise ValueError(f"VIVA row missing {sorted(required - row.keys())}")
        identifier = f"viva:{row['index']}"
        path = local_media(media_root, str(row["image_file"]))
        files[identifier] = path
        actions = list(row["action_list"])
        answer = str(row["answer"]).strip()
        action_by_letter = {str(a).split(".", 1)[0].strip(): str(a).split(".", 1)[1].strip()
                            for a in actions if "." in str(a)}
        if len(action_by_letter) != len(actions) or answer not in action_by_letter:
            raise ValueError("Malformed VIVA action list or answer.")
        parts = [Part(type="image", path=str(path))]
        choices: list[tuple[str, int]] = []
        if task == "values":
            parts.append(Part(type="text", text="Proposed action: " + action_by_letter[answer]))
            if not isinstance(row.get("values"), dict) or not {"positive", "negative"} <= row["values"].keys():
                raise ValueError("VIVA values task requires published positive and negative candidates.")
            for key, relevance in (("positive", 1), ("negative", 0)):
                for value in row["values"][key]:
                    surface = str(value).split(":", 1)[0].strip()
                    if not surface:
                        raise ValueError("Empty VIVA value candidate.")
                    choices.append((surface, relevance))
        else:
            choices = [(text, int(letter == answer)) for letter, text in action_by_letter.items()]
        if len({surface.casefold() for surface, _ in choices}) != len(choices):
            raise ValueError("VIVA has duplicate or conflicting candidate surfaces within a sample.")
        # Candidate IDs/order do not reveal their positive/negative annotation.
        choices.sort(key=lambda choice: hashlib.sha256(choice[0].encode()).hexdigest())
        anchors = []
        for surface, relevance in choices:
            anchor_id = hashlib.sha256(surface.encode()).hexdigest()[:16]
            anchors.append(Anchor(id=anchor_id, surface=surface, language="en"))
            records.append({"sample_id": identifier, "anchor_id": anchor_id, "relevant": relevance})
        candidates[identifier] = tuple(anchors)
        samples.append(Sample(id=identifier, source="viva", language="en", group_id=str(row["image_file"]),
                              parts=tuple(parts)))
    labels = pd.DataFrame(records, columns=["sample_id", "anchor_id", "relevant"]).set_index(
        ["sample_id", "anchor_id"])
    manifest = build_manifest(
        "viva", [annotation_path], samples, seed=seed, expected_sha256=expected_sha256,
        source_url="https://huggingface.co/datasets/zhehuderek/VIVA_Benchmark_EMNLP24",
        license_note="Dataset card: CC BY-NC-SA 4.0; media source rights remain applicable.",
        label_kind="action_conditioned_local_value_candidates" if task == "values" else "action_choice",
        candidate_scope="per_instance", negative_meaning="irrelevant_or_contradictory",
        media=_availability(files, require_media), benchmark_split_note="HF train is packaging, not a research split.",
        excluded_input_fields=["reason", "situation_description", "action_answer", "value_explanations"],
    )
    return DatasetBundle(samples, labels, manifest, candidates)


def load_vatex(
    annotation_path: str | Path, media_root: str | Path, *, language: str = "en",
    split: str | None = None, videos_are_clips: bool = True, extension: str = ".mp4",
    include_captions: bool = True, require_media: bool = True, seed: int = 42,
    expected_sha256: Mapping[str, str] | None = None,
) -> DatasetBundle:
    """Native VATEX videoID=YouTubeID_Start_End; video/captions are separate samples.

    Clipped media uses videoID.mp4 with no second time crop. Full-video mode uses
    YouTubeID.mp4 and passes original temporal boundaries to the video processor.
    """
    if language not in {"en", "zh"}:
        raise ValueError("VATEX provides en and zh caption fields.")
    data = read_json(annotation_path)
    if not isinstance(data, list):
        raise ValueError("VATEX annotations must be a JSON list.")
    samples, records, files = [], [], {}
    for row in data:
        if "videoID" not in row:
            raise ValueError("VATEX annotation is missing videoID.")
        clip_id = str(row["videoID"])
        try:
            youtube_id, start, end = clip_id.rsplit("_", 2)
            start, end = float(start), float(end)
        except (ValueError, TypeError) as exc:
            raise ValueError("Use native VATEX IDs ending in _StartTime_EndTime.") from exc
        if not youtube_id or start < 0 or end <= start:
            raise ValueError("Invalid VATEX temporal boundaries.")
        path = local_media(media_root, (clip_id if videos_are_clips else youtube_id) + extension)
        identifier = f"vatex:{clip_id}:video"
        files[identifier] = path
        part = Part(type="video", path=str(path), clip_start=None if videos_are_clips else start,
                    clip_end=None if videos_are_clips else end)
        samples.append(Sample(id=identifier, source="vatex", group_id=youtube_id,
                              parts=(part,), metadata={"source_start": start, "source_end": end}))
        if include_captions:
            captions = row.get("enCap" if language == "en" else "chCap", [])
            if not isinstance(captions, list):
                raise ValueError("VATEX captions must be lists.")
            for i, caption in enumerate(captions):
                caption_id = f"vatex:{clip_id}:{language}:{i}"
                samples.append(Sample(id=caption_id, source="vatex", language=language,
                                      group_id=youtube_id, pair_id=identifier,
                                      parts=(Part(type="text", text=str(caption)),)))
                records.append({"video_id": identifier, "caption_id": caption_id, "relevant": 1})
    labels = pd.DataFrame(records, columns=["video_id", "caption_id", "relevant"])
    manifest = build_manifest(
        "vatex", [annotation_path], samples, seed=seed, expected_sha256=expected_sha256,
        official_splits={s.id: split for s in samples} if split is not None else None,
        source_url="https://eric-xw.github.io/vatex-website/download.html",
        license_note="Annotations CC BY 4.0; upstream does not redistribute YouTube videos.",
        label_kind="video_caption_relevance_not_value", media=_availability(files, require_media),
        videos_are_clips=videos_are_clips, caption_language=language,
    )
    return DatasetBundle(samples, labels, manifest)
