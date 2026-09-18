"""VLanchor: conditional anchor measurement, with explicit provenance."""

from .types import (
    Anchor, Bridge, CalibrationArtifact, MeasurementMatrix, ModelSpec,
    Part, ResourceProfile, Sample, ScoreTable, StudySpec, TargetSpan,
)
from .engine import measure, score
from .calibration import fit_reference, transform, to_matrix
from .analysis import cluster, compare_groups, semantic_shift, semantic_network
from .reliability import audit_reliability
from .optimization import optimize_bridges

__version__ = "0.1.0"

__all__ = [
    "Anchor", "Bridge", "CalibrationArtifact", "MeasurementMatrix", "ModelSpec",
    "Part", "ResourceProfile", "Sample", "ScoreTable", "StudySpec", "TargetSpan",
    "measure", "score", "fit_reference", "transform", "to_matrix", "cluster",
    "compare_groups", "semantic_shift", "semantic_network", "audit_reliability",
    "optimize_bridges",
]
