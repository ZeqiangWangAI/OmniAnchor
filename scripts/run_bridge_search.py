"""Train-only bounded PMPO search using the existing optimizer and local Qwen rewriting."""
import argparse
import gc
import json
import os
import shutil
from pathlib import Path
from time import perf_counter

import pandas as pd

from omnianchor import fit_reference, measure, to_matrix, transform
from omnianchor.backends import HFBackend
from omnianchor.campaign import append_event, create_run
from omnianchor.io import load_samples, load_scores, load_spec, read_json, save_calibration, save_matrix, save_scores, write_json
from omnianchor.optimization import OptimizationConfig, OptimizationData, optimize_bridges
from omnianchor.provenance import file_hash
from omnianchor.types import Bridge, ModelSpec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--objective", choices=["reference_guided", "validity", "reliability", "alpha", "random"], default="reference_guided")
    parser.add_argument("--seed-cache", type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Search requires a Slurm allocation.")
    root = Path(__file__).resolve().parents[1]
    spec = load_spec(root / "configs/studies/valueeval_fixed3.json")
    seeds = [Bridge.model_validate(r) for r in read_json(root / "configs/bridges/expresses_value_seed8.json")]
    reference = load_samples(args.data / "reference/samples.json")
    search = load_samples(args.data / "search/samples.json")
    if any(s.metadata.get("split") != "train" for s in reference + search):
        raise ValueError("Reference and search both require explicit train split.")
    labels = pd.read_csv(args.data / "search/labels.csv", index_col="sample_id")
    labels.index = labels.index.astype(str)
    data = OptimizationData(labels.loc[[s.id for s in search], [a.id for a in spec.anchors]].to_numpy(),
                            tuple(s.id for s in search), tuple(a.id for a in spec.anchors),
                            group_ids=tuple(s.group_id for s in search),
                            reference_group_ids=tuple(s.group_id for s in reference))
    data.validate()
    config = OptimizationConfig(forbidden_surfaces=tuple(a.surface for a in spec.anchors), objective=args.objective)
    create_run(args.output, {"purpose": "train_only_bridge_search", "config": vars(config),
                             "spec": spec.model_dump(), "slurm_job_id": os.environ["SLURM_JOB_ID"],
                             "reference_ids": [s.id for s in reference], "search_ids": [s.id for s in search],
                             "source_hashes": {str(p.relative_to(root)): file_hash(p) for folder in ["src", "scripts", "configs"]
                                               for p in (root / folder).rglob("*") if p.is_file() and "__pycache__" not in str(p)},
                             "data_hashes": {str(p.relative_to(args.data)): file_hash(p) for role in ["reference", "search"]
                                             for p in (args.data / role).glob("*")},
                             "human_semantic_review": "pending; no dev/test evaluation in this driver",
                             "seed_cache": str(args.seed_cache) if args.seed_cache else None,
                             "budget_note": "16 logical distinct scoring attempts; reused common seeds are recorded separately from physical model calls"})
    import torch
    torch.manual_seed(42)
    backend = None
    fatal = None

    def unload():
        nonlocal backend
        backend = None
        gc.collect()
        torch.cuda.empty_cache()

    def score_bridge(bridge):
        nonlocal backend, fatal
        if fatal is not None:
            raise RuntimeError("Previous scoring failure; no further model calls.")
        folder = args.output / "candidates" / bridge.id
        folder.mkdir(parents=True, exist_ok=False)
        write_json(folder / "bridge.json", bridge)
        started = perf_counter()
        try:
            if args.seed_cache and bridge.id in {b.id for b in seeds}:
                source = args.seed_cache / bridge.id
                if read_json(source / "bridge.json") != bridge.model_dump():
                    raise ValueError("Cached seed wording differs.")
                ref, raw = [load_scores(source / name) for name in ["reference.parquet", "search.parquet"]]
                expected_identity = HFBackend(spec.model, spec.resources, shared_prefill=False).identity
                for table, expected_samples in [(ref, reference), (raw, search)]:
                    if table.manifest["instrument"]["model"] != expected_identity or table.manifest["event"] != spec.event:
                        raise ValueError("Cached seed backend identity differs.")
                    if table.manifest["anchors"] != [a.model_dump() for a in spec.anchors] or table.manifest["bridges"] != [bridge.model_dump()]:
                        raise ValueError("Cached seed coordinates differ.")
                    observed = {s["id"]: {k: v for k, v in s.items() if k != "content_hash"} for s in table.manifest["samples"]}
                    if observed != {s.id: s.model_dump(mode="json") for s in expected_samples}:
                        raise ValueError("Cached seed inputs differ.")
                    if table.manifest["execution"]["failed_items"]:
                        raise ValueError("Cached seed contains failures.")
                calibration = fit_reference(ref)
                matrix = to_matrix(transform(raw, calibration), variant="reference_z", missing="error")
                copied = {}
                for name in ["reference.parquet", "reference.parquet.manifest.json", "search.parquet", "search.parquet.manifest.json",
                             "calibration.json", "search-z.npz", "search-z.npz.manifest.json"]:
                    path = source / name
                    if path.is_file():
                        shutil.copy2(path, folder / name)
                        copied[name] = file_hash(path)
                write_json(folder / "reuse.json", {"source": str(source), "sha256": copied, "physical_model_calls": 0})
                append_event(args.output, "seed_reused", bridge_id=bridge.id, source=str(source))
                return matrix.values[[matrix.sample_ids.index(s.id) for s in search]][:,
                    [matrix.anchor_ids.index(a.id) for a in spec.anchors]]
            if backend is None:
                backend = HFBackend(spec.model, spec.resources, shared_prefill=False)
            candidate_spec = spec.model_copy(update={"bridges": (bridge,)})
            ref = measure(reference, candidate_spec, backend=backend)
            save_scores(ref, folder / "reference.parquet")
            if ref.manifest["execution"]["failed_items"]:
                raise RuntimeError("Reference scoring failed; stop rather than discard a difficult sample.")
            calibration = fit_reference(ref)
            save_calibration(calibration, folder / "calibration.json")
            raw = measure(search, candidate_spec, backend=backend)
            save_scores(raw, folder / "search.parquet")
            if raw.manifest["execution"]["failed_items"]:
                raise RuntimeError("Search scoring failed; stop and preserve failures.")
            matrix = to_matrix(transform(raw, calibration), variant="reference_z", missing="error")
            save_matrix(matrix, folder / "search-z.npz")
            indices = [matrix.sample_ids.index(s.id) for s in search]
            columns = [matrix.anchor_ids.index(a.id) for a in spec.anchors]
            append_event(args.output, "candidate_scored", bridge_id=bridge.id,
                         seconds=perf_counter() - started)
            return matrix.values[indices][:, columns]
        except Exception as exc:
            fatal = repr(exc)
            append_event(args.output, "candidate_failed", bridge_id=bridge.id, error=repr(exc))
            raise

    def operator(selected, count, round_seed):
        if fatal is not None:
            return []
        unload()
        models = read_json(root / "configs/models-20260910.json")["models"]
        model_id = "Qwen/Qwen3-VL-4B-Instruct"
        generator = HFBackend(ModelSpec(id=model_id, revision=models[model_id]), spec.resources)
        generator._ensure_loaded()
        prompt = ("Rewrite measurement bridge prefixes while preserving exactly this relation: "
                  "a value expressed by the supplied material, not endorsement or approval. "
                  "Do not insert example values, answers, example materials or special control tokens. "
                  f"Return only a JSON array of {count} distinct English strings. "
                  "Each string must end with a colon followed by an escaped newline. "
                  "Current prefixes: " + json.dumps([b.prefix for b in selected]))
        messages = [{"role": "user", "content": prompt}]
        text = generator._processor.apply_chat_template(messages, tokenize=False,
                                                        add_generation_prompt=True, enable_thinking=False)
        inputs = generator._processor(text=[text], return_tensors="pt", padding=False, truncation=False).to("cuda")
        torch.manual_seed(42)
        with torch.inference_mode():
            output = generator._model.generate(**inputs, do_sample=True, temperature=.7,
                                                top_p=.9, max_new_tokens=128)
        token_ids = output[0, inputs["input_ids"].shape[1]:].tolist()
        decoded = generator._processor.tokenizer.decode(token_ids, skip_special_tokens=True)
        folder = args.output / f"rewrite-{round_seed}"
        folder.mkdir(exist_ok=False)
        write_json(folder / "generation.json", {"model": generator.identity, "prompt": prompt,
                                                "seed": 42, "round_seed": round_seed, "temperature": .7,
                                                "top_p": .9, "max_new_tokens": 128,
                                                "token_ids": token_ids, "decoded": decoded})
        del generator, inputs, output
        gc.collect()
        torch.cuda.empty_cache()
        proposals = json.loads(decoded)
        if not isinstance(proposals, list) or len(proposals) != count or not all(isinstance(p, str) and p.endswith(":\n") for p in proposals):
            raise ValueError("Rewrite is not the required exact JSON bridge array; preserve incumbent.")
        return [Bridge(id=f"rewrite_{round_seed}_{i}", prefix=p, relation="expresses_value") for i, p in enumerate(proposals)]

    try:
        artifact = optimize_bridges(seeds, data, config, score_bridge=score_bridge, operator=operator)
        write_json(args.output / "selected.json", artifact)
        if fatal is not None:
            raise RuntimeError(fatal)
        append_event(args.output, "search_completed", artifact_status=artifact.status,
                     human_semantic_review="pending", test_evaluations=0)
    except BaseException as exc:
        append_event(args.output, "failed", error=repr(exc))
        raise
    finally:
        unload()


if __name__ == "__main__":
    main()
