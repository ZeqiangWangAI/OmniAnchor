"""Rerun the frozen32 FMAT contexts with contextual single-token alignment audits."""
import argparse
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from omnianchor.baselines import HFSingleTokenMLM
from omnianchor.campaign import append_event, create_run
from omnianchor.io import read_json, write_json
from omnianchor.provenance import file_hash, runtime_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", choices=["bert", "roberta"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Require Slurm CUDA; no CPU fallback.")
    root = Path(__file__).resolve().parents[1]
    model_id = {"bert": "google-bert/bert-base-uncased", "roberta": "FacebookAI/roberta-base"}[args.model]
    revision = read_json(root / "configs/models-20260910.json")["models"][model_id]
    frame = pd.read_csv(args.data)
    frame = frame[frame.model == model_id.split("/")[-1]]
    create_run(args.output, {"model_id": model_id, "revision": revision, "data_sha256": file_hash(args.data),
        "runtime": runtime_manifest(), "slurm_job_id": os.environ["SLURM_JOB_ID"], "seed": 42,
        "precision": "float32 for masked replication", "max_tokens": 512, "batch_size": 1,
        "event": "single masked token with both left and right context; not causal anchor score",
        "source_sha256": file_hash(Path(__file__)), "direction": "Male logp minus Female logp"})
    start = perf_counter()
    try:
        model = HFSingleTokenMLM(model_id, revision, batch_size=1, local_files_only=False)
        model._load("AutoModelForMaskedLM")
        torch, tokenizer = model._torch, model._tokenizer
        torch.manual_seed(42)
        write_json(args.output / "hardware.json", {"gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda})
        torch.cuda.reset_peak_memory_stats()
        rows, audit = [], []
        for _, row in frame.iterrows():
            template = row["query"].replace("{TARGET}", row.T_word)
            masked = template.replace("[MASK]", tokenizer.mask_token)
            filled = template.replace("[MASK]", row.M_word)
            ids = tokenizer.encode(masked)
            position = ids.index(tokenizer.mask_token_id)
            filled_ids = tokenizer.encode(filled)
            # Derive the context-correct token, including RoBERTa's leading-space BPE.
            if len(ids) != len(filled_ids) or ids[:position] != filled_ids[:position] or ids[position+1:] != filled_ids[position+1:]:
                raise ValueError(f"Non-single-token or nonlocal replacement: {filled}")
            target_id = filled_ids[position]
            surface = tokenizer.decode([target_id], clean_up_tokenization_spaces=False)
            if tokenizer.encode(surface, add_special_tokens=False) != [target_id]:
                raise ValueError("Decoded contextual token fails single-token round trip.")
            value = float(model.score([masked], [surface])[0, 0])
            result = {"qid": int(row.qid), "target": row.T_word, "gender": row.MASK,
                "word": row.M_word, "logp": value, "stored_prob": float(row.prob),
                "stored_logp": float(np.log(row.prob)), "logp_difference": value-float(np.log(row.prob))}
            rows.append(result)
            audit.append({**result, "masked_text": masked, "filled_text": filled,
                "input_ids": ids, "filled_ids": filled_ids, "mask_position": position,
                "candidate_surface": surface, "candidate_token_id": target_id, "alignment": "exact_single_replacement"})
            append_event(args.output, "score_completed", **result)
        pd.DataFrame(rows).to_csv(args.output / "scores.csv", index=False)
        write_json(args.output / "token-alignment-audit.json", audit)
        write_json(args.output / "summary.json", {"score_count": len(rows), "contexts": len(rows)//2,
            "max_abs_logp_difference_from_stored": max(abs(r["logp_difference"]) for r in rows),
            "interpretation": "checkpoint/preprocessing replication comparison; no equality assumed"})
        write_json(args.output / "cost.json", {"total_seconds": perf_counter()-start,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated()})
        append_event(args.output, "completed")
    except BaseException as exc:
        append_event(args.output, "failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
