"""Shared public contracts. Coordinates retain exact surfaces and provenance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

Event = Literal["token_prefix", "turn_terminated"]
Variant = Literal["raw_logp", "mean_token_logp", "reference_log_ratio", "reference_z"]


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Part(FrozenModel):
    type: Literal["text", "image", "video"]
    text: str | None = None
    path: str | None = None
    clip_start: float | None = Field(default=None, ge=0)
    clip_end: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_content(self):
        if self.type == "text":
            if self.text is None or self.path is not None:
                raise ValueError("Text parts require text and no media path.")
        elif not self.path or self.text is not None:
            raise ValueError("Media parts require a path and no text.")
        if self.type != "video" and (self.clip_start is not None or self.clip_end is not None):
            raise ValueError("Clip boundaries are only valid for video.")
        if self.clip_end is not None and self.clip_end <= (self.clip_start or 0):
            raise ValueError("clip_end must follow clip_start.")
        return self


class TargetSpan(FrozenModel):
    part_index: int = Field(ge=0)
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    lemma: str = Field(min_length=1)


class Sample(FrozenModel):
    id: str = Field(min_length=1)
    parts: tuple[Part, ...] = Field(min_length=1)
    language: str | None = None
    source: str | None = None
    group_id: str | None = None
    time: str | None = None
    pair_id: str | None = None
    target: TargetSpan | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_target(self):
        if self.target is not None:
            t = self.target
            if t.part_index >= len(self.parts):
                raise ValueError("Target part does not exist.")
            part = self.parts[t.part_index]
            if part.type != "text" or not (t.start < t.end <= len(part.text or "")):
                raise ValueError("Target span must index Unicode characters in a text part.")
        return self


class Anchor(FrozenModel):
    id: str = Field(min_length=1)
    surface: str = Field(min_length=1)
    language: str = "en"
    concept_id: str | None = None

    @model_validator(mode="after")
    def nonblank(self):
        if not self.surface.strip():
            raise ValueError("An anchor must contain non-whitespace text.")
        return self


class Bridge(FrozenModel):
    id: str = Field(min_length=1)
    prefix: str = Field(min_length=1)
    relation: str = "associated_with"
    language: str = "en"


class ModelSpec(FrozenModel):
    backend: Literal["hf", "toy"] = "hf"
    id: str = "Qwen/Qwen3.5-4B"
    revision: str = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    device: str = "cuda"
    precision: Literal["bf16", "fp32"] = "bf16"
    attention_implementation: str | None = None
    adapter: str | None = None
    """Scoring adapter name from ``backends.adapters``; None selects by ``config.model_type``."""
    turn_end_token: str | None = None
    """Exact turn-end token for ``turn_terminated``; required when the adapter declares none."""
    chat_template_kwargs: dict[str, Any] | None = None
    """Overrides the adapter's chat-template keyword arguments when set."""


class ImageBudget(FrozenModel):
    min_pixels: int = Field(default=65536, gt=0)
    max_pixels: int = Field(default=262144, gt=0)

    @model_validator(mode="after")
    def bounds(self):
        if self.min_pixels > self.max_pixels:
            raise ValueError("min_pixels exceeds max_pixels")
        return self


class VideoBudget(FrozenModel):
    sampled_frames: int = Field(default=8, gt=0)
    sampling: Literal["uniform"] = "uniform"
    max_pixels_per_frame: int = Field(default=65536, gt=0)


class Limits(FrozenModel):
    input_text_tokens: int = Field(default=2048, gt=0)
    anchor_continuation_tokens: int = Field(default=32, gt=0)
    total_postprocessor_tokens: int = Field(default=8192, gt=0)
    implicit_truncation: Literal[False] = False


class ResourceProfile(FrozenModel):
    image: ImageBudget = Field(default_factory=ImageBudget)
    video: VideoBudget = Field(default_factory=VideoBudget)
    limits: Limits = Field(default_factory=Limits)


class StudySpec(FrozenModel):
    name: str = "study"
    model: ModelSpec = Field(default_factory=ModelSpec)
    anchors: tuple[Anchor, ...] = Field(min_length=1)
    bridges: tuple[Bridge, ...] = Field(min_length=1)
    event: Event = "token_prefix"
    resources: ResourceProfile = Field(default_factory=ResourceProfile)
    seed: int = 42
    include_token_details: bool = False
    system_prompt: str | None = None

    @model_validator(mode="after")
    def coherent_coordinates(self):
        for label, values in (("anchor", self.anchors), ("bridge", self.bridges)):
            if len({v.id for v in values}) != len(values):
                raise ValueError(f"Duplicate {label} IDs.")
        if len({(b.relation, b.language) for b in self.bridges}) != 1:
            raise ValueError("One study must use one bridge relation and language.")
        return self


@dataclass
class ScoreTable:
    frame: pd.DataFrame
    manifest: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.frame)


@dataclass
class CalibrationArtifact:
    statistics: pd.DataFrame
    manifest: dict[str, Any] = field(default_factory=dict)


@dataclass
class MeasurementMatrix:
    values: np.ndarray
    sample_ids: tuple[str, ...]
    anchor_ids: tuple[str, ...]
    variant: str = "raw_logp"
    manifest: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=np.float64)
        if self.values.shape != (len(self.sample_ids), len(self.anchor_ids)):
            raise ValueError("Matrix shape does not match IDs.")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("Duplicate sample IDs.")
        if len(set(self.anchor_ids)) != len(self.anchor_ids):
            raise ValueError("Duplicate anchor IDs.")
