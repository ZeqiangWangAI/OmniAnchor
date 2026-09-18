"""Target-aware WiC and DWUG readers, preserving original Unicode offsets."""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path
from typing import Mapping

import pandas as pd

from ..types import Anchor, Part, Sample, TargetSpan
from .common import DatasetBundle, build_manifest, require_columns


def general_anchors(language: str = "en") -> tuple[Anchor, ...]:
    """Legacy/default 24-concept fixture bank, not the train-derived general128 pack."""
    banks = {
        "en": ("person", "animal", "plant", "place", "object", "substance", "motion", "change",
               "communication", "knowledge", "emotion", "relationship", "work", "money", "power",
               "law", "health", "food", "art", "nature", "technology", "time", "space", "quantity"),
        "zh": ("人物", "动物", "植物", "地点", "物体", "物质", "运动", "变化", "交流", "知识",
               "情感", "关系", "工作", "金钱", "权力", "法律", "健康", "食物", "艺术", "自然",
               "技术", "时间", "空间", "数量"),
    }
    if language not in banks:
        raise ValueError("Provide explicit anchors for languages other than en/zh.")
    return tuple(Anchor(id=f"general:{i:02d}", surface=text, language=language,
                        concept_id=f"general:{i:02d}") for i, text in enumerate(banks[language]))


def build_general_anchors(bundle: DatasetBundle, n: int = 128) -> tuple[Anchor, ...]:
    """Rank exact WiC train lemmas by pair frequency, then Unicode surface order.

    Each pair contributes once regardless of its number of usage records. No labels
    or context words are read. IDs depend only on the lemma, so packs of different
    sizes share identical prefixes. The caller must preserve bundle.manifest and
    the serialized pack together; this function does not mutate source provenance.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("Anchor count n must be a positive integer.")
    if bundle.manifest.get("dataset") != "wic":
        raise ValueError("General anchors require a WiC dataset bundle.")
    splits = bundle.manifest.get("splits", {})
    if (bundle.manifest.get("split_strategy") != "official_preserved" or not bundle.samples
            or set(splits) != {sample.id for sample in bundle.samples}
            or any(splits[sample.id] != "train" or sample.metadata.get("split") != "train"
                   for sample in bundle.samples)):
        raise ValueError("General anchors require only explicitly declared WiC train samples.")
    pair_lemmas = {}
    for sample in bundle.samples:
        if (sample.source != "wic" or sample.language != "en" or not sample.pair_id
                or sample.target is None or not sample.target.lemma.strip()):
            raise ValueError("WiC samples require English target lemmas and pair IDs.")
        lemma = sample.target.lemma
        if sample.pair_id in pair_lemmas and pair_lemmas[sample.pair_id] != lemma:
            raise ValueError("A WiC pair has inconsistent target lemmas.")
        pair_lemmas[sample.pair_id] = lemma
    counts = Counter(pair_lemmas.values())
    if len(counts) < n:
        raise ValueError(f"Requested {n} anchors but WiC train contains only {len(counts)} unique lemmas.")
    surfaces = sorted(counts, key=lambda surface: (-counts[surface], surface))[:n]
    return tuple(Anchor(id=f"general:wic:en:{surface}", surface=surface, language="en",
                        concept_id=f"general:wic:en:{surface}") for surface in surfaces)


def load_wic(
    data_path: str | Path, gold_path: str | Path | None = None, *, split: str | None = None,
    seed: int = 42, expected_sha256: Mapping[str, str] | None = None,
) -> DatasetBundle:
    """Original 5-column tab-delimited WiC file; target indices are whitespace token indices."""
    lines = Path(data_path).read_text(encoding="utf-8").splitlines()
    samples, rows = [], []
    gold = None
    if gold_path is not None:
        gold = Path(gold_path).read_text(encoding="utf-8").splitlines()
        if len(gold) != len(lines) or any(value.strip() not in {"T", "F"} for value in gold):
            raise ValueError("WiC gold must contain one T/F label per data row.")
    for i, line in enumerate(lines):
        fields = line.split("\t")
        if len(fields) != 5:
            raise ValueError(f"WiC row {i} must contain five tab-delimited fields.")
        lemma, pos, indices, text1, text2 = fields
        if not re.fullmatch(r"\d+-\d+", indices):
            raise ValueError("WiC target positions must have index1-index2 syntax.")
        pair_id = f"wic:{split or Path(data_path).stem}:{i}"
        ids = []
        for side, text, token_index in zip((1, 2), (text1, text2), map(int, indices.split("-")), strict=True):
            spans = list(re.finditer(r"\S+", text))
            if token_index >= len(spans):
                raise ValueError(f"WiC target token index out of bounds in row {i}.")
            span = spans[token_index]
            identifier = f"{pair_id}:{side}"
            ids.append(identifier)
            samples.append(Sample(id=identifier, source="wic", language="en", group_id=lemma,
                                  pair_id=pair_id, parts=(Part(type="text", text=text),),
                                  target=TargetSpan(part_index=0, start=span.start(), end=span.end(), lemma=lemma),
                                  metadata={"pos": pos}))
        record = {"pair_id": pair_id, "sample1_id": ids[0], "sample2_id": ids[1]}
        if gold is not None:
            record["same_sense"] = int(gold[i].strip() == "T")
        rows.append(record)
    labels = pd.DataFrame(rows).set_index("pair_id") if rows else pd.DataFrame()
    manifest = build_manifest(
        "wic", [data_path] + ([gold_path] if gold_path else []), samples, seed=seed,
        expected_sha256=expected_sha256,
        official_splits={s.id: split for s in samples} if split is not None else None,
        source_url="https://pilehvar.github.io/wic/", license_note="WiC: CC BY-NC 4.0.",
        label_kind="pair_same_sense", target_offset_unit="unicode_character",
    )
    return DatasetBundle(samples, labels, manifest)


def load_dwug(
    uses_path: str | Path, judgments_path: str | Path | None = None, *,
    language: str = "en", seed: int = 42,
    expected_sha256: Mapping[str, str] | None = None,
) -> DatasetBundle:
    """A published WUG uses.csv is TAB-delimited despite its .csv suffix.

    Modern indexes_target_token and older indexes_target spellings are supported;
    discontinuous target spans are rejected rather than silently shortened.
    """
    # Published WUG TSVs contain literal quotes in contexts, not CSV quote escaping.
    data = pd.read_csv(uses_path, sep="\t", keep_default_na=False, quoting=csv.QUOTE_NONE)
    span_column = "indexes_target_token" if "indexes_target_token" in data else "indexes_target"
    require_columns(data, ["identifier", "lemma", "context", "grouping", span_column], "DWUG uses")
    samples = []
    for row in data.to_dict("records"):
        value = str(row[span_column])
        if not re.fullmatch(r"\d+:\d+", value):
            raise ValueError(f"Unsupported DWUG target span {value!r}; expected one start:end range.")
        start, end = map(int, value.split(":"))
        samples.append(Sample(id=str(row["identifier"]), language=language, source="dwug",
                              group_id=str(row["lemma"]), time=str(row["grouping"]),
                              parts=(Part(type="text", text=str(row["context"])),),
                              target=TargetSpan(part_index=0, start=start, end=end, lemma=str(row["lemma"]))))
    labels, paths = pd.DataFrame(), [uses_path]
    if judgments_path is not None:
        labels = pd.read_csv(judgments_path, sep="\t", keep_default_na=False, quoting=csv.QUOTE_NONE)
        require_columns(labels, ["identifier1", "identifier2", "judgment"], "DWUG judgments")
        ids = {s.id for s in samples}
        if (set(labels["identifier1"].astype(str)) | set(labels["identifier2"].astype(str))) - ids:
            raise ValueError("DWUG judgments reference uses absent from uses_path.")
        labels["judgment"] = pd.to_numeric(labels["judgment"])
        if not labels["judgment"].isin([0, 1, 2, 3, 4]).all():
            raise ValueError("Expected DWUG judgments 1–4 or 0 for cannot decide.")
        # Keep 0 as missing/undecidable annotation; evaluation must exclude it explicitly.
        labels["is_decidable"] = labels["judgment"] != 0
        paths.append(judgments_path)
    manifest = build_manifest(
        "dwug", paths, samples, seed=seed, expected_sha256=expected_sha256,
        source_url="https://www.ims.uni-stuttgart.de/en/research/resources/experiment-data/wugs/",
        license_note="Version-specific data terms; some releases prohibit redistribution.",
        label_kind="human_use_pair_relatedness", undecidable_label=0,
        target_offset_unit="unicode_character", evaluation_group="lemma",
        split_note="Hash split is a local grouped partition, not an official SemEval training split.",
    )
    return DatasetBundle(samples, labels, manifest)
