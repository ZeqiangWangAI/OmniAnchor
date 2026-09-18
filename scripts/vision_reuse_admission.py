"""Require completed numerical evidence and the exact tested vision implementation."""
from pathlib import Path

from omnianchor.io import read_json
from omnianchor.provenance import file_hash


def admit_vision_reuse(summary: Path) -> dict:
    evidence = read_json(summary)
    source = summary.parent.parent / "source/src/omnianchor/backends/vision_reuse.py"
    current = Path(__file__).resolve().parents[1] / "src/omnianchor/backends/vision_reuse.py"
    if evidence["status"] != "passed" or len(evidence["comparisons"]) != 120:
        raise ValueError("Require complete three-model vision reuse gate.")
    if any(not row["passed"] for row in evidence["comparisons"]):
        raise ValueError("Vision reuse gate contains a failed comparison.")
    if evidence["native_verification"] != {"qwen35": "passed", "qwen3vl": "passed"}:
        raise ValueError("Native numerical/state checks did not pass.")
    if file_hash(source) != file_hash(current):
        raise ValueError("Vision reuse source differs from the tested implementation.")
    return {"gate_summary": str(summary), "gate_summary_sha256": file_hash(summary),
            "tested_source_sha256": file_hash(source), "shared_prefill": False,
            "language_decoder_cache": False, "scope": "one_candidate_call"}
