from pathlib import Path

import pytest

from scripts.frozen_evaluation import validate_frozen_evaluation
from vlanchor.io import write_json
from vlanchor.provenance import file_hash


def test_final_contract_checks_method_role_and_exact_input_bytes(tmp_path):
    config, samples, verification, contract = [tmp_path / n for n in
        ["config.json", "samples.json", "verification.json", "contract.json"]]
    row = {"id": "protected", "parts": [{"type": "text", "text": "fixed input"}], "metadata": {"split": "test"}}
    write_json(config, {"frozen": True})
    write_json(samples, [row])
    write_json(verification, [{**row, "metadata": {"split": "train"}}])
    root = Path(__file__).resolve().parents[1]
    definition = {"status": "frozen", "methods": ["native"], "sample_ids": ["protected"],
        "samples_sha256": file_hash(samples), "config_sha256": file_hash(config),
        "media_sha256": {"protected": []}, "verification_samples": str(verification),
        "verification_samples_sha256": file_hash(verification),
        "scoring_source_sha256": {"src/vlanchor/engine.py": file_hash(root / "src/vlanchor/engine.py")}}
    write_json(contract, definition)
    assert validate_frozen_evaluation(contract, samples, config, "native")["sha256"] == file_hash(contract)
    with pytest.raises(ValueError, match="Method"):
        validate_frozen_evaluation(contract, samples, config, "qwen-reranker")
    write_json(samples, [{**row, "metadata": {"split": "dev"}}])
    with pytest.raises(ValueError, match="bytes"):
        validate_frozen_evaluation(contract, samples, config, "native")
    write_json(contract, {**definition, "samples_sha256": file_hash(samples)})
    with pytest.raises(ValueError, match="test population"):
        validate_frozen_evaluation(contract, samples, config, "native")
