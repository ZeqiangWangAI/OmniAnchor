"""Adapters for published text tables. Labels remain outside the scored text."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from ..types import Part, Sample
from .common import DatasetBundle, build_manifest, normalized_group, read_table, require_columns


VALUEEVAL_LABELS = (
    "Self-direction: thought", "Self-direction: action", "Stimulation", "Hedonism",
    "Achievement", "Power: dominance", "Power: resources", "Face", "Security: personal",
    "Security: societal", "Tradition", "Conformity: rules", "Conformity: interpersonal",
    "Humility", "Benevolence: caring", "Benevolence: dependability", "Universalism: concern",
    "Universalism: nature", "Universalism: tolerance", "Universalism: objectivity",
)


def load_valueeval(
    arguments_path: str | Path, labels_path: str | Path | None = None, *,
    split: str | None = None, language: str = "en", seed: int = 42,
    expected_sha256: Mapping[str, str] | None = None,
) -> DatasetBundle:
    """Touché23 TSV files; pass their published split explicitly to preserve it."""
    data = read_table(arguments_path)
    require_columns(data, ["Argument ID", "Premise", "Stance", "Conclusion"], "ValueEval")
    samples = []
    for row in data.to_dict("records"):
        premise, stance, conclusion = (str(row[k]) for k in ("Premise", "Stance", "Conclusion"))
        samples.append(Sample(
            id=str(row["Argument ID"]), language=language, source="valueeval",
            group_id=normalized_group(conclusion),
            parts=(Part(type="text", text=f"Premise: {premise}\nStance: {stance}\nConclusion: {conclusion}"),),
        ))
    paths = [arguments_path]
    labels = pd.DataFrame(index=pd.Index([s.id for s in samples], name="sample_id"))
    if labels_path is not None:
        gold = read_table(labels_path)
        require_columns(gold, ["Argument ID", *VALUEEVAL_LABELS], "ValueEval labels")
        gold["Argument ID"] = gold["Argument ID"].astype(str)
        if gold["Argument ID"].duplicated().any():
            raise ValueError("Duplicate ValueEval label IDs.")
        labels = gold.set_index("Argument ID").loc[:, list(VALUEEVAL_LABELS)]
        if set(labels.index) != {s.id for s in samples}:
            raise ValueError("ValueEval arguments and labels have different ID sets.")
        labels = labels.loc[[s.id for s in samples]].apply(pd.to_numeric)
        if not np.isin(labels.to_numpy(), [0, 1]).all():
            raise ValueError("ValueEval labels must be binary.")
        labels.index.name = "sample_id"
        paths.append(labels_path)
    manifest = build_manifest(
        "valueeval", paths, samples, seed=seed, expected_sha256=expected_sha256,
        official_splits={s.id: split for s in samples} if split is not None else None,
        source_url="https://zenodo.org/records/10564870",
        license_note="Check exact release: Zenodo README states CC BY-SA 4.0; HF card differs.",
        label_kind="human_value_multilabel", label_columns=list(VALUEEVAL_LABELS),
        canonical_text_fields=["Premise", "Stance", "Conclusion"],
    )
    return DatasetBundle(samples, labels, manifest)


def load_emobank(
    path: str | Path, *, rating_perspective: str = "combined",
    individual_ratings_path: str | Path | None = None, ratings_path: str | Path | None = None,
    seed: int = 42, expected_sha256: Mapping[str, str] | None = None,
) -> DatasetBundle:
    """Join official text/splits to combined, reader, or writer ratings.

    emobank.csv is a weighted reader+writer average. Reader/writer requires its
    explicit published aggregate file or individual ratings file. Raw aggregation
    follows upstream aggregation.ipynb: remove all-1 triples, retain N>1, round 2.
    """
    if rating_perspective not in {"combined", "reader", "writer"}:
        raise ValueError("rating_perspective must be combined, reader, or writer.")
    if individual_ratings_path is not None and ratings_path is not None:
        raise ValueError("Supply either individual ratings or aggregate ratings, not both.")
    if rating_perspective == "combined" and (individual_ratings_path or ratings_path):
        raise ValueError("Additional ratings require an explicit reader or writer perspective.")
    if rating_perspective != "combined" and not (individual_ratings_path or ratings_path):
        raise ValueError("Reader/writer perspective requires its corresponding ratings file.")
    data = read_table(path)
    require_columns(data, ["id", "split", "V", "A", "D", "text"], "EmoBank emobank.csv")
    data["id"] = data["id"].astype(str)
    if data["id"].duplicated().any():
        raise ValueError("EmoBank metadata contains duplicate IDs.")
    paths, excluded, unmatched, rating_counts = [path], [], [], {}
    if rating_perspective != "combined":
        ratings_file = individual_ratings_path or ratings_path
        ratings = read_table(ratings_file)
        require_columns(ratings, ["id", "V", "A", "D"], "EmoBank perspective ratings")
        ratings["id"] = ratings["id"].astype(str)
        ratings[["V", "A", "D"]] = ratings[["V", "A", "D"]].apply(pd.to_numeric)
        values = ratings[["V", "A", "D"]].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not ((values >= 1) & (values <= 5)).all():
            raise ValueError("EmoBank individual/aggregate VAD ratings must lie in [1, 5].")
        if individual_ratings_path is not None:
            ratings = ratings.loc[~(ratings[["V", "A", "D"]] == 1).all(axis=1)]
            counts = ratings.groupby("id").size()
            means = ratings.groupby("id")[["V", "A", "D"]].mean().round(2)
            means = means.loc[counts > 1]
            rating_counts = {str(k): int(v) for k, v in counts.loc[counts > 1].items()}
        else:
            if ratings["id"].duplicated().any():
                raise ValueError("Published aggregate perspective ratings must have unique IDs.")
            means = ratings.set_index("id")[["V", "A", "D"]]
        excluded = sorted(set(data["id"]) - set(means.index))
        unmatched = sorted(set(means.index) - set(data["id"]))
        data = data.loc[data["id"].isin(means.index)].copy()
        data[["V", "A", "D"]] = means.loc[data["id"]].to_numpy()
        if data.empty:
            raise ValueError("No valid perspective ratings match the supplied official text/splits.")
        paths.append(ratings_file)
    def document_group(identifier, text):
        fields = str(identifier).rsplit("_", 2)
        return fields[0] if len(fields) == 3 and all(x.isdigit() for x in fields[1:]) else normalized_group(text)
    samples = [Sample(id=str(r["id"]), language="en", source="emobank",
                      group_id=document_group(r["id"], str(r["text"])),
                      metadata={"rating_perspective": rating_perspective},
                      parts=(Part(type="text", text=str(r["text"])),))
               for r in data.to_dict("records")]
    labels = data.set_index(data["id"].astype(str))[["V", "A", "D"]].apply(pd.to_numeric)
    labels.index.name = "sample_id"
    if not np.isfinite(labels.to_numpy()).all() or not ((labels >= 1) & (labels <= 5)).all().all():
        raise ValueError("EmoBank ratings must lie in [1, 5].")
    manifest = build_manifest(
        "emobank", paths, samples, seed=seed, expected_sha256=expected_sha256,
        official_splits=dict(zip([s.id for s in samples], data["split"], strict=True)),
        source_url="https://github.com/JULIELab/EmoBank", license_note="See upstream license.",
        label_kind="affect_VAD_not_value", rating_range=[1, 5],
        rating_perspective=rating_perspective,
        aggregation="drop_all_one_triples; N>1; mean; round_2" if individual_ratings_path else "published_aggregate",
        excluded_without_valid_perspective_ratings=excluded,
        ratings_without_official_metadata=unmatched, valid_rating_counts=rating_counts,
        group_rule="source_document_prefix_when_id_has_offsets; otherwise normalized_text",
    )
    return DatasetBundle(samples, labels, manifest)


def load_chinese_emobank(
    path: str | Path, *, text_column: str | None = None, seed: int = 42,
    expected_sha256: Mapping[str, str] | None = None,
) -> DatasetBundle:
    """Published CVAW/CVAP/CVAS/CVAT tables, including XLSX releases."""
    data = read_table(path)
    if text_column is None:
        columns = [c for c in ("Word", "Phrase", "Sentence", "Text") if c in data]
        if len(columns) != 1:
            raise ValueError("Specify text_column: expected one of Word/Phrase/Sentence/Text.")
        text_column = columns[0]
    require_columns(data, [text_column, "Valence_Mean", "Arousal_Mean"], "Chinese EmoBank")
    samples = [Sample(id=f"chinese_emobank:{i}", language="zh", source="chinese_emobank",
                      group_id=normalized_group(str(r[text_column])),
                      parts=(Part(type="text", text=str(r[text_column])),))
               for i, r in enumerate(data.to_dict("records"))]
    labels = data[["Valence_Mean", "Arousal_Mean"]].apply(pd.to_numeric).rename(
        columns={"Valence_Mean": "V", "Arousal_Mean": "A"})
    labels.index = pd.Index([s.id for s in samples], name="sample_id")
    if not np.isfinite(labels.to_numpy()).all() or not ((labels >= 1) & (labels <= 9)).all().all():
        raise ValueError("Chinese EmoBank VA means must lie in [1, 9].")
    manifest = build_manifest(
        "chinese_emobank", [path], samples, seed=seed, expected_sha256=expected_sha256,
        source_url="https://github.com/NYCU-NLP/Chinese-EmoBank",
        license_note="Upstream repository does not clearly specify a data license; acquire locally.",
        label_kind="affect_VA_not_value", rating_range=[1, 9], text_column=text_column,
    )
    return DatasetBundle(samples, labels, manifest)


def load_fmat(
    path: str | Path, *, object_name: str | None = None, seed: int = 42,
    expected_sha256: Mapping[str, str] | None = None,
) -> DatasetBundle:
    """Import original FMAT stored model probabilities, not human value labels.

    CSV/TSV exports require query, prob, model and token (or M_word). RData requires
    an explicit object such as d1a and optional pyreadr; arbitrary R code is never run.
    """
    path = Path(path)
    if path.suffix.lower() in {".rdata", ".rda"}:
        if not object_name:
            raise ValueError("RData imports require an explicit object_name, e.g. d1a.")
        try:
            import pyreadr
        except ImportError as exc:
            raise ImportError("Install omnianchor[data] for pyreadr, or supply a CSV export.") from exc
        objects = pyreadr.read_r(str(path), use_objects=[object_name])
        if object_name not in objects:
            raise ValueError(f"RData does not contain dataframe {object_name!r}.")
        data = objects[object_name]
    else:
        data = read_table(path)
    require_columns(data, ["query", "prob", "model"], "FMAT stored probabilities")
    token_col = "M_word" if "M_word" in data else "token"
    require_columns(data, [token_col], "FMAT stored probabilities")
    probabilities = pd.to_numeric(data["prob"]).to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Stored FMAT probabilities must be finite and in [0, 1].")
    samples, records = [], []
    for i, row in enumerate(data.to_dict("records")):
        identifier = f"fmat:{object_name or path.stem}:{i}"
        query = str(row["query"])
        group = str(row.get("T_word", row.get("TARGET", query)))
        samples.append(Sample(id=identifier, language="en", source="fmat",
                              group_id=normalized_group(group),
                              parts=(Part(type="text", text=query),)))
        records.append({"sample_id": identifier, "model": str(row["model"]),
                        "token": str(row[token_col]), "prob": probabilities[i]})
    labels = pd.DataFrame(records, columns=["sample_id", "model", "token", "prob"]).set_index("sample_id")
    manifest = build_manifest(
        "fmat", [path], samples, seed=seed, expected_sha256=expected_sha256,
        source_url="https://osf.io/5e2hr/", license_note="OSF data: CC BY 4.0.",
        license_api="https://api.osf.io/v2/licenses/563c1cf88c5e4a3877f9e96a/",
        label_kind="stored_model_probabilities_not_human_labels", object_name=object_name,
    )
    return DatasetBundle(samples, labels, manifest)
