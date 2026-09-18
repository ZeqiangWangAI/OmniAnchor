"""Score the example samples with every alias in a model list and run the native acceptance check.

Software verification only: synthetic stimuli, no human criterion, no accuracy claim.
Each model is loaded, scored (text, turn-terminated event, one synthetic image), verified
against the independent full-logit forward, then released before the next one.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

from omnianchor import Anchor, Bridge, ModelSpec, Part, Sample, StudySpec, measure
from omnianchor.backends import HFBackend
from omnianchor.config import model_spec_from_alias
from omnianchor.io import load_samples, save_scores, write_json
from omnianchor.provenance import runtime_manifest
from omnianchor.verification import verify_native

ANCHORS = tuple(Anchor(id=a.replace(" ", "_"), surface=a)
                for a in ["freedom", "red", "blue circle", "social justice"])
BRIDGES = (Bridge(id="b1", prefix="A concept associated with this material is:\n"),
           Bridge(id="b2", prefix="This material is associated with the concept of:\n"))


def image_fixture(directory: Path) -> Sample:
    directory.mkdir(parents=True, exist_ok=True)
    picture = Image.new("RGB", (256, 256), "white")
    ImageDraw.Draw(picture).rectangle((48, 48, 208, 208), fill="red")
    path = directory / "red_square.png"
    picture.save(path)
    return Sample(id="image-red_square", parts=(Part(type="image", path=str(path)),),
                  source="synthetic_smoke", metadata={"split": "test"})


def smoke_model(alias: str, samples: list[Sample], image: Sample, output: Path) -> dict:
    import torch
    output.mkdir(parents=True, exist_ok=True)
    model_spec = ModelSpec.model_validate(model_spec_from_alias(alias))
    spec = StudySpec(name=f"smoke-{alias}", model=model_spec, anchors=ANCHORS, bridges=BRIDGES,
                     include_token_details=True)
    backend = HFBackend(model_spec, resources=spec.resources)
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    text = measure(samples, spec, backend=backend, cache=output / "score-cache")
    save_scores(text, output / "scores-text.parquet")
    failures = text.frame[text.frame.status.ne("ok")]
    if len(failures):
        raise RuntimeError(f"Text scoring failures: {failures.to_dict('records')}")
    replay = measure(samples, spec, backend=backend, cache=output / "score-cache")
    if replay.frame.raw_logp.tolist() != text.frame.raw_logp.tolist():
        raise AssertionError("Cached and uncached scores differ.")
    terminated = backend.score_candidates(samples[0], BRIDGES[0], ANCHORS[:1], event="turn_terminated")[0]
    prefix = text.frame[(text.frame.sample_id == samples[0].id) & (text.frame.bridge_id == "b1")
                        & (text.frame.anchor_id == ANCHORS[0].id)].iloc[0]
    if terminated["status"] != "ok":
        raise AssertionError(f"turn_terminated failed: {terminated}")
    if terminated["raw_logp"] > prefix.raw_logp + 0.01:
        raise AssertionError("Terminated event cannot exceed prefix probability.")
    adapter = backend.adapter
    media = measure([image], spec, backend=backend)
    save_scores(media, output / "scores-image.parquet")
    media_ok = bool(media.frame.status.eq("ok").all())
    if "image" in adapter.modalities and not media_ok:
        raise RuntimeError(f"Image scoring failed: {media.frame.to_dict('records')}")
    if "image" not in adapter.modalities and media_ok:
        raise AssertionError("A text-only adapter must not report image scores as ok.")
    # Some published chat templates prepend a long system preamble; 1024 tokens of full
    # logits is still a bounded, explicit materialisation (vocab x tokens x float32).
    native = verify_native(backend, output=output / "native_verification.json",
                           bridge=BRIDGES[0], anchors=(ANCHORS[0], ANCHORS[2]),
                           image_sample=image if "image" in adapter.modalities else None,
                           max_full_sequence_tokens=1024)
    if native["status"] != "passed":
        raise AssertionError("Native full-logits verification failed; see native_verification.json.")
    report = {
        "status": "passed", "alias": alias, "adapter": adapter.name, "adapter_kind": adapter.kind,
        "model": backend.identity, "elapsed_seconds": time.perf_counter() - started,
        "score_items_text": len(text), "cache_hits_replay": replay.manifest["execution"]["cache_hits"],
        "turn_terminated": {"raw_logp": terminated["raw_logp"], "token_count": terminated["token_count"],
                            "prefix_raw_logp": float(prefix.raw_logp)},
        "image": {"supported": "image" in adapter.modalities,
                  "statuses": sorted(set(media.frame.status)),
                  "errors": sorted(set(media.frame.get("error", []).dropna())) if not media_ok else []},
        "native_verification": {k: native["checks"][k].get("passed", native["checks"][k].get("source"))
                                for k in native["checks"]},
        "raw_logp_range": [float(text.frame.raw_logp.min()), float(text.frame.raw_logp.max())],
        "prefix_token_count_last": backend.last_preparation.get("prefix_token_count"),
        "verification_sequence_lengths": native["checks"]["selected_vs_full"]["sequence_lengths"],
        "peak_memory_bytes": torch.cuda.max_memory_allocated(), "gpu": torch.cuda.get_device_name(),
    }
    write_json(output / "report.json", report)
    del backend
    torch.cuda.empty_cache()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True, help="YAML with a `models:` list of aliases")
    parser.add_argument("--samples", required=True, help="Text samples JSON (examples/samples.json)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    aliases = yaml.safe_load(Path(args.models).read_text(encoding="utf-8"))["models"]
    output = Path(args.output)
    samples = load_samples(args.samples)
    image = image_fixture(output / "stimuli")
    summary = {"purpose": "multi_model_software_smoke_not_construct_validation",
               "runtime": runtime_manifest(), "models": {}}
    for alias in aliases:
        try:
            summary["models"][alias] = smoke_model(alias, samples, image, output / alias)
        except Exception as exc:  # noqa: BLE001 - one model must not hide the others' evidence
            summary["models"][alias] = {"status": "failed", "alias": alias, "error": str(exc),
                                        "traceback": traceback.format_exc()}
        print(json.dumps({k: v for k, v in summary["models"][alias].items() if k != "traceback"},
                         default=str), flush=True)
        write_json(output / "models_summary.json", summary)
    summary["status"] = "passed" if all(m["status"] == "passed" for m in summary["models"].values()) else "failed"
    write_json(output / "models_summary.json", summary)
    print(json.dumps({alias: m["status"] for alias, m in summary["models"].items()}), flush=True)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
