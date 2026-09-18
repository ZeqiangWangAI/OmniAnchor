"""Named causal full-sentence PLC versus anchor-only diagnostic on frozen FMAT32."""
import argparse
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from vlanchor.backends.hf import HFBackend, PreparedCandidate, reference_token_logps
from vlanchor.campaign import append_event, create_run
from vlanchor.io import load_spec, write_json
from vlanchor.provenance import file_hash, runtime_manifest
from vlanchor.types import Anchor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Require Slurm CUDA.")
    spec = load_spec(args.config)
    frame = pd.read_csv(args.data)
    frame = frame[frame.model == "bert-base-uncased"]
    create_run(args.output, {"purpose": "E1 diagnostic, not a causal replication of masked probabilities",
        "spec": spec.model_dump(), "data_sha256": file_hash(args.data), "runtime": runtime_manifest(),
        "source_sha256": file_hash(Path(__file__)), "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "event": "full sentence completion under fixed 'Complete the sentence.' user chat condition, no EOT, no length normalization",
        "partition": "prefix + contextual gender token(s) + suffix; whitespace may attach to anchor BPE",
        "shared_prefill": False, "seed": 42, "batch_size": 1})
    start = perf_counter()
    try:
        import torch
        torch.manual_seed(42)
        backend = HFBackend(spec.model, spec.resources, shared_prefill=False)
        backend._ensure_loaded()
        tokenizer = backend._processor.tokenizer
        prompt = backend._processor.apply_chat_template([
            {"role": "user", "content": [{"type": "text", "text": "Complete the sentence."}]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prefix_ids = tokenizer.encode(prompt, add_special_tokens=False)
        write_json(args.output / "condition.json", {"text": prompt, "token_ids": prefix_ids,
            "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda})
        torch.cuda.reset_peak_memory_stats()
        rows, audit = [], []
        for _, row in frame.iterrows():
            template = row["query"].replace("{TARGET}", row.T_word)
            left, right = template.split("[MASK]")
            sentence = left + row.M_word + right
            text = prompt + sentence
            encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
            if ids[:len(prefix_ids)] != prefix_ids:
                raise ValueError("Chat/completion token boundary is incompatible.")
            a0, a1 = len(prompt)+len(left), len(prompt)+len(left)+len(row.M_word)
            positions = [i for i, (lo, hi) in enumerate(offsets) if lo < a1 and hi > a0]
            if not positions or positions != list(range(positions[0], positions[-1]+1)):
                raise ValueError("Anchor token span is not contiguous.")
            lo, hi = positions[0], positions[-1]+1
            for i in positions:
                begin, end = offsets[i]
                outside = text[begin:min(end, a0)] + text[max(begin, a1):end]
                if outside.strip():
                    raise ValueError("Anchor BPE crosses a non-whitespace semantic boundary.")
            if lo < len(prefix_ids):
                raise ValueError("Anchor overlaps the fixed condition.")

            def score(stop, prefix):
                inputs = {"input_ids": torch.tensor([ids[:stop]], device="cuda"),
                          "attention_mask": torch.ones((1, stop), dtype=torch.long, device="cuda")}
                item = PreparedCandidate(inputs, prefix, Anchor(id="gender", surface=row.M_word), "prefix", hi-lo)
                return reference_token_logps(backend._model, item)

            all_logps = np.asarray(score(len(ids), len(prefix_ids)))
            p, a = lo-len(prefix_ids), hi-len(prefix_ids)
            anchor_logps = np.asarray(score(hi, lo))
            error = float(np.max(np.abs(anchor_logps-all_logps[p:a])))
            if error > 0.01:
                raise ValueError(f"Suffix causal invariance error {error} exceeds0.01nat.")
            record = {"qid": int(row.qid), "target": row.T_word, "gender": row.MASK,
                "sentence": sentence, "prefix_logp": float(all_logps[:p].sum()),
                "anchor_logp": float(all_logps[p:a].sum()), "suffix_logp": float(all_logps[a:].sum()),
                "plc_full_sentence_logp": float(all_logps.sum()), "anchor_token_count": hi-lo,
                "sentence_token_count": len(all_logps), "suffix_invariance_max_abs_nat": error}
            rows.append(record)
            audit.append({**record, "input_ids": ids, "condition_length": len(prefix_ids),
                "anchor_start_token": lo, "anchor_end_token": hi, "token_logps": all_logps.tolist(),
                "anchor_only_token_logps": anchor_logps.tolist(), "offsets": offsets})
            append_event(args.output, "score_completed", **record)
        pd.DataFrame(rows).to_csv(args.output / "scores.csv", index=False)
        write_json(args.output / "token-alignment-audit.json", audit)
        write_json(args.output / "cost.json", {"total_seconds": perf_counter()-start,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "scored_sentences": len(rows)})
        append_event(args.output, "completed")
    except BaseException as exc:
        append_event(args.output, "failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
