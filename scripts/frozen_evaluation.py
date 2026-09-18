"""Validate an immutable final-evaluation contract without changing split labels."""
from pathlib import Path

from vlanchor.io import load_samples,read_json
from vlanchor.provenance import file_hash


def validate_frozen_evaluation(contract_path: Path, samples_path: Path, config_path: Path, method: str) -> dict:
    contract=read_json(contract_path)
    root=Path(__file__).resolve().parents[1]
    if contract["status"]!="frozen" or method not in contract["methods"]:
        raise ValueError("Method is not in a frozen evaluation protocol.")
    if file_hash(samples_path)!=contract["samples_sha256"] or file_hash(config_path)!=contract["config_sha256"]:
        raise ValueError("Protected input/config bytes differ from the frozen protocol.")
    samples=load_samples(samples_path)
    if [s.id for s in samples]!=contract["sample_ids"] or any(s.metadata["split"]!="test" for s in samples):
        raise ValueError("Require the exact frozen test population, preserving its role.")
    media={s.id:[{"part_index":i,"sha256":file_hash(p.path)} for i,p in enumerate(s.parts) if p.path]
           for s in samples}
    if media!=contract["media_sha256"]:
        raise ValueError("Protected media bytes changed after freeze.")
    if file_hash(root/contract["verification_samples"])!=contract["verification_samples_sha256"]:
        raise ValueError("Development verification inputs changed after freeze.")
    for name,digest in contract["scoring_source_sha256"].items():
        if file_hash(root/name)!=digest:
            raise ValueError(f"Scoring source changed after final freeze: {name}")
    return {"contract":str(contract_path),"sha256":file_hash(contract_path),"protocol":contract}
