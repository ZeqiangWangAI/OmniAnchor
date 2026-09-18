"""Freeze the bounded E3 materials and one-variable instrument contrasts without labels."""
import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path

from vlanchor.campaign import select_smoke
from vlanchor.io import load_samples, load_spec, read_json, write_json
from vlanchor.provenance import file_hash
from vlanchor.types import Anchor


ALIASES = dict(joy="happiness", sadness="sorrow", fear="fearfulness", anger="rage",
    calmness="tranquility", excitement="thrill", pleasure="enjoyment", discomfort="unease",
    confidence="self-assurance", helplessness="powerlessness", interest="curiosity", boredom="tedium")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=False)
    paths = {
        "emobank": {"config": "configs/smoke/qwen35-en.json",
            "train": "data/prepared/emobank-affect-20260910-01/probe_train/samples.json",
            "dev": "data/prepared/emobank-affect-20260910-01/dev/samples.json",
            "test": "data/prepared/emobank-final-20260910-01/samples.json",
            "reference": "data/prepared/emobank-affect-20260910-01/reference/samples.json"},
        "dwug": {"config": "configs/studies/dwug_general128.json",
            "smoke": "data/prepared/dwug-target-inputs-20260910-01/smoke32.json",
            "train": "data/prepared/dwug-target-inputs-20260910-01/train.json",
            "dev": "data/prepared/dwug-target-inputs-20260910-01/dev.json",
            "test": "data/prepared/dwug-target-inputs-20260910-01/test.json",
            "reference": "data/prepared/dwug-target-inputs-20260910-01/reference64.json"},
    }
    models = read_json(root/"configs/models-20260910.json")["models"]
    bank_path = root/"data/prepared/wic-anchors-20260910/general256_en.json"
    full_bank = tuple(Anchor.model_validate(a) for a in read_json(bank_path)["anchors"])
    if len(full_bank) != 256:
        raise ValueError("Require the original frozen WiC train bank.")
    records, files = {}, {}
    for study, names in paths.items():
        source = {name: root/path for name, path in names.items()}
        spec = load_spec(source["config"])
        inputs = {role: load_samples(source[role]) for role in ["train", "dev", "test", "reference"]}
        for role, samples in inputs.items():
            required = "train" if role == "reference" else role
            if not samples or any(s.metadata.get("split") != required for s in samples):
                raise ValueError("Unexpected split role.")
            if any(p.type != "text" for s in samples for p in s.parts):
                raise ValueError("The bounded sensitivity study uses text only.")
        selected = sorted(inputs["test"], key=lambda s: hashlib.sha256(f"42\0{s.id}".encode()).hexdigest())[:128]
        if len(selected) != 128 or len({s.id for s in selected}) != 128:
            raise ValueError("Require 128 unique protected materials per study.")
        smoke = (load_samples(source["smoke"]) if "smoke" in source else
                 select_smoke(inputs["train"]+inputs["dev"], n_per_split=16))
        if len(smoke) != 32 or any(s.metadata.get("split") not in {"train", "dev"} for s in smoke):
            raise ValueError("Require exactly 32 train/dev technical verification inputs.")
        if {s.id for s in inputs["reference"]} & {s.id for s in smoke+selected}:
            raise ValueError("Reference overlaps sensitivity rows.")
        folder = args.output/study
        write_json(folder/"test128.json", selected)
        write_json(folder/"smoke32.json", smoke)
        write_json(folder/"reference.json", inputs["reference"])
        variants = {"base": spec,
            "terminated": spec.model_copy(update={"event": "turn_terminated"}),
            "second-model": spec.model_copy(update={"model": spec.model.model_copy(update={
                "id": "Qwen/Qwen3-VL-4B-Instruct", "revision": models["Qwen/Qwen3-VL-4B-Instruct"]})})}
        duplicate = spec.anchors[0].model_copy(update={"id": "duplicate:"+spec.anchors[0].id})
        unrelated = tuple(Anchor(id="control:"+s, surface=s, language="en") for s in ["hexadecimal", "carburetor", "stalactite"])
        variants["augmented"] = spec.model_copy(update={"anchors": spec.anchors+(duplicate,)+unrelated})
        if study == "emobank":
            if set(ALIASES) != {a.id for a in spec.anchors}:
                raise ValueError("Alias map does not match original affect12 coordinates.")
            variants["aliases"] = spec.model_copy(update={"anchors": tuple(
                a.model_copy(update={"surface": ALIASES[a.id], "concept_id": a.id}) for a in spec.anchors)})
            variants["phrases"] = spec.model_copy(update={"anchors": tuple(
                a.model_copy(update={"surface": "a feeling of "+a.surface, "concept_id": a.id}) for a in spec.anchors)})
        else:
            if spec.anchors != full_bank[:128]:
                raise ValueError("DWUG128 is not the original WiC256 prefix.")
            variants["general256"] = spec.model_copy(update={"anchors": full_bank})
        for name, variant in variants.items():
            write_json(folder/f"{name}.json", variant)
        records[study] = {"source_sha256": {name: file_hash(path) for name, path in source.items()},
            "test_ids": [s.id for s in selected], "smoke_ids": [s.id for s in smoke],
            "reference_ids": [s.id for s in inputs["reference"]],
            "variants": {name: {"anchors": len(v.anchors), "bridges": len(v.bridges), "event": v.event,
                "model": v.model.id} for name, v in variants.items()},
            "cross_split_source_groups": sorted({s.group_id for s in inputs["train"]+inputs["dev"]} & {s.group_id for s in selected})}
        files.update({str(p.relative_to(args.output)): file_hash(p) for p in folder.glob("*.json")})
    write_json(args.output/"manifest.json", {"status": "frozen", "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Bounded E3 instrument sensitivity; no labels read or changes fed back to primary studies",
        "generator_sha256": file_hash(Path(__file__)), "seed": 42, "total_test_materials": 256,
        "selection": "Within each study first128 ascending SHA256(42 NUL sample_id), before inspecting corresponding test scores",
        "timing": "Hash rule previously specified; exact IDs and contrasts frozen now. Earlier FMAT/OASIS results already seen; no claim of prospective registration before all research.",
        "studies": records, "files_sha256": files, "general256_source_sha256": file_hash(bank_path),
        "fixed_controls": "Same material, resources, three bridges, BF16 batch1, uncached decoder; each variant changes one instrument property; no factorial expansion",
        "calibration": "Fit each changed instrument to the same named train reference, never to evaluation; 64/128/256 use prefixes of general256 with per-coordinate reference retained",
        "alias_scope": "Intended related emotion labels, not proven interchangeable constructs: rage/thrill/curiosity can alter intensity or meaning. Report per-coordinate sensitivity without asserting semantic equivalence or clinical validity.",
        "length_scope": "Bare emotion vs a feeling of emotion; lexical framing and length covary. Report event token counts/raw/mean-token/z; no pure causal length claim.",
        "augmentation_scope": "Duplicate first anchor adds geometric weight; three unrelated controls were chosen without test labels. No extra validity signal presumed.",
        "analysis": "All fixed and single-bridge raw/logratio/z scores; coordinate rank agreement, pair-distance rank agreement, token counts, z/translation invariants and template stability. Paired group-bootstrap1000 intervals are descriptive sensitivity; no optimization, no new superiority family.",
        "budget": "11 configurations, each32train/dev smoke then128test and originalreference; reuse exact compatible primary raw scores where possible. General64/128 derived from256. Full launch only after measured smoke GPU/storage cost.",
        "hardware": "Slurm RTX5000Ada for both checkpoints; no CPU neural fallback, no budget truncation",
        "test_admission": "This inventory is not a scoring admission. Freeze each final scoring contract after32technical verification and before executing test; preserve split=test."})


if __name__ == "__main__":
    main()
