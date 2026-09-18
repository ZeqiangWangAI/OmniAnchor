"""Verify complete score grids, calibration and translation invariance; no test labels."""
import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np

from vlanchor import fit_reference, to_matrix, transform
from vlanchor.campaign import create_run, merge_score_shards, append_event
from vlanchor.io import load_samples, load_scores, read_json, save_calibration, save_matrix, save_scores, write_json
from vlanchor.types import ScoreTable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.run / "exit_code.txt").read_text().strip() != "0":
        raise ValueError("Cannot summarize an incomplete or failed smoke as passed.")
    create_run(args.output, {"purpose": "software_and_cost_validation_not_construct_validity", "source_run": str(args.run)})
    reports = {}
    for folder in sorted(args.run.glob("qwen*")):
        language = folder.name.rsplit("-", 1)[1]
        dataset = "emobank-reader" if language == "en" else "chinese-emobank-sentence"
        samples = load_samples(args.data / dataset / "smoke-samples.json")
        raw = merge_score_shards([load_scores(p) for p in sorted(folder.glob("part-*.parquet"))], [s.id for s in samples])
        if raw.manifest["execution"]["failed_items"]:
            raise ValueError("Failed score rows remain.")
        output = args.output / folder.name
        save_scores(raw, output / "scores.parquet")
        ids = {s.id for s in samples if s.metadata["split"] == "train"}
        metadata = deepcopy(raw.manifest)
        metadata["samples"] = [s for s in metadata["samples"] if s["id"] in ids]
        ref = ScoreTable(raw.frame[raw.frame.sample_id.isin(ids)].copy(), metadata)
        calibration = fit_reference(ref)
        save_calibration(calibration, output / "calibration.json")
        adjusted = transform(raw, calibration)
        matrices = {v: to_matrix(adjusted, variant=v) for v in ["raw_logp", "reference_log_ratio", "reference_z"]}
        for variant, matrix in matrices.items():
            if not np.isfinite(matrix.values).all():
                raise ValueError("Calibration has undefined coordinates.")
            save_matrix(matrix, output / f"{variant}.npz")
        x, y = matrices["raw_logp"].values, matrices["reference_log_ratio"].values
        error = float(np.max(np.abs((x - x[0]) - (y - y[0]))))
        if error > 1e-10:
            raise ValueError("Fixed reference translation invariance failed.")
        verification = read_json(folder / "native-verification.json")
        if verification["status"] != "passed":
            raise ValueError("Native verification failed.")
        reports[folder.name] = {"samples": len(samples), "score_items": len(raw.frame),
                                "reference_samples": len(ids), "failed_items": 0,
                                "max_translation_error": error, "native_verification": "passed",
                                "cost": read_json(folder / "cost.json")}
    if len(reports) != 4:
        raise ValueError("Expected both models and both languages.")
    write_json(args.output / "summary.json", reports)
    append_event(args.output, "completed")
    print({k: {f: v[f] for f in ["samples", "score_items", "failed_items", "max_translation_error"]} for k, v in reports.items()})


if __name__ == "__main__":
    main()
