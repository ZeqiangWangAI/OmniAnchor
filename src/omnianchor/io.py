"""Portable artifact storage; JSON metadata travels with every numeric artifact."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .types import CalibrationArtifact, MeasurementMatrix, Sample, ScoreTable, StudySpec


def jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump(mode="json"))
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, pd.DataFrame):
        return jsonable(value.to_dict(orient="records"))
    if isinstance(value, (np.ndarray, pd.Series)):
        return jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".omnianchor-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload + "\n")
        os.replace(name, path)
    finally:
        if Path(name).exists():
            Path(name).unlink()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_spec(path: str | Path) -> StudySpec:
    return StudySpec.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def load_samples(path: str | Path) -> list[Sample]:
    path = Path(path)
    if path.suffix == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
    else:
        records = read_json(path)
    samples = []
    for row in records:
        sample = Sample.model_validate(row)
        parts = tuple(part.model_copy(update={"path": str((path.parent / part.path).resolve())})
                      if part.path and not Path(part.path).is_absolute() else part
                      for part in sample.parts)
        samples.append(sample.model_copy(update={"parts": parts}))
    if len({s.id for s in samples}) != len(samples):
        raise ValueError("Duplicate sample IDs in input.")
    return samples


def save_scores(table: ScoreTable, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.frame.to_parquet(path, index=False)
    write_json(path.with_suffix(path.suffix + ".manifest.json"), table.manifest)


def load_scores(path: str | Path) -> ScoreTable:
    path = Path(path)
    return ScoreTable(pd.read_parquet(path), read_json(path.with_suffix(path.suffix + ".manifest.json")))


def save_calibration(artifact: CalibrationArtifact, path: str | Path) -> None:
    write_json(path, {"statistics": artifact.statistics, "manifest": artifact.manifest})


def load_calibration(path: str | Path) -> CalibrationArtifact:
    value = read_json(path)
    return CalibrationArtifact(pd.DataFrame(value["statistics"]), value["manifest"])


def save_matrix(matrix: MeasurementMatrix, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".npz":
        np.savez_compressed(path, values=matrix.values, sample_ids=np.array(matrix.sample_ids),
                            anchor_ids=np.array(matrix.anchor_ids))
    elif path.suffix == ".parquet":
        frame = pd.DataFrame(matrix.values, columns=matrix.anchor_ids)
        frame.insert(0, "sample_id", matrix.sample_ids)
        frame.to_parquet(path, index=False)
    else:
        raise ValueError("Matrix output must use .npz or .parquet.")
    write_json(path.with_suffix(path.suffix + ".manifest.json"),
               {"variant": matrix.variant, "manifest": matrix.manifest,
                "sample_ids": matrix.sample_ids, "anchor_ids": matrix.anchor_ids})


def load_matrix(path: str | Path) -> MeasurementMatrix:
    path = Path(path)
    meta = read_json(path.with_suffix(path.suffix + ".manifest.json"))
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            values = data["values"]
            sample_ids = tuple(data["sample_ids"].tolist())
            anchor_ids = tuple(data["anchor_ids"].tolist())
    elif path.suffix == ".parquet":
        frame = pd.read_parquet(path)
        sample_ids = tuple(frame.pop("sample_id"))
        anchor_ids = tuple(frame.columns)
        values = frame.to_numpy()
    else:
        raise ValueError("Matrix input must use .npz or .parquet.")
    if list(sample_ids) != meta["sample_ids"] or list(anchor_ids) != meta["anchor_ids"]:
        raise ValueError("Matrix coordinate IDs differ from its manifest.")
    return MeasurementMatrix(values, sample_ids, anchor_ids, meta["variant"], meta["manifest"])
