"""Bounded software demonstration; synthetic stimuli are not scientific gold labels."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import platform
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw

from omnianchor import (Anchor, Bridge, ModelSpec, Part, Sample, StudySpec, fit_reference,
                      measure, to_matrix, transform)
from omnianchor.analysis import cluster, pca, semantic_network, semantic_shift
from omnianchor.backends import HFBackend, ToyBackend
from omnianchor.io import save_calibration, save_matrix, save_scores, write_json
from omnianchor.provenance import runtime_manifest
from omnianchor.reliability import audit_reliability, template_radii
from omnianchor.types import ScoreTable


def fixtures(directory: Path, *, video: bool = True):
    directory.mkdir(parents=True, exist_ok=True)
    for name, shape, color in [("red_square", "rectangle", "red"), ("blue_circle", "ellipse", "blue")]:
        picture = Image.new("RGB", (256, 256), "white")
        getattr(ImageDraw.Draw(picture), shape)((48, 48, 208, 208), fill=color)
        picture.save(directory / f"{name}.png")
    texts = [
        "People should be free to choose their beliefs and express their opinions.",
        "Neighbors cooperate to share food and care for vulnerable people.",
        "The locked gate and emergency rules protect the community.",
        "The painting contains a red square on a white background.",
        "Everyone deserves fair treatment and equal opportunities.",
        "A child draws a blue circle and shows it to a friend.",
    ]
    samples = [Sample(id=f"text-{i}", parts=(Part(type="text", text=t),), language="en",
                      source="synthetic_demo", group_id=f"document-{i}",
                      time="early" if i < 3 else "late", metadata={"split": "train", "target_id": "demo"})
               for i, t in enumerate(texts)]
    for name in ["red_square", "blue_circle"]:
        samples.append(Sample(id=f"image-{name}", parts=(Part(type="image", path=str(directory/f"{name}.png")),),
                              source="synthetic_demo", metadata={"split": "test"}))
    samples.append(Sample(id="image-text", parts=(
        Part(type="image", path=str(directory/"red_square.png")),
        Part(type="text", text="This sign marks a community meeting about freedom.")),
        language="en", source="synthetic_demo", metadata={"split": "test"}))
    if video:
        import av
        path = directory/"moving_circle.mp4"
        with av.open(str(path), "w") as container:
            stream = container.add_stream("mpeg4", rate=4)
            stream.width = stream.height = 256
            stream.pix_fmt = "yuv420p"
            for i in range(8):
                frame = Image.new("RGB", (256, 256), "white")
                ImageDraw.Draw(frame).ellipse((16+i*16, 96, 80+i*16, 160), fill="blue")
                for packet in stream.encode(av.VideoFrame.from_image(frame)):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        samples.append(Sample(id="video", parts=(Part(type="video", path=str(path)),),
                              source="synthetic_demo", metadata={"split": "test"}))
    chinese = [Sample(id=f"zh-{i}", language="zh-Hans", source="synthetic_demo",
                      parts=(Part(type="text", text=t),), metadata={"split": "test"})
               for i,t in enumerate(["每个人都应当有表达意见的自由。", "邻居互相帮助，照顾需要支持的人。", "画面中有一个蓝色圆形。"])]
    return samples, chinese


def subset(table: ScoreTable, ids: set[str]) -> ScoreTable:
    manifest = deepcopy(table.manifest)
    manifest["samples"] = [s for s in manifest["samples"] if s["id"] in ids]
    return ScoreTable(table.frame[table.frame.sample_id.isin(ids)].copy(), manifest)


def run(output: Path, backend_kind: str, *, video=True):
    output.mkdir(parents=True, exist_ok=True)
    samples, chinese = fixtures(output/"stimuli", video=video)
    anchors = tuple(Anchor(id=a.replace(" ", "_"), surface=a) for a in ["freedom", "red", "blue circle", "social justice"])
    bridges = (Bridge(id="b1", prefix="A concept associated with this material is:\n"),
               Bridge(id="b2", prefix="This material is associated with the concept of:\n"))
    model_spec = ModelSpec() if backend_kind == "hf" else ModelSpec(backend="toy", id="toy", revision="v1", device="cpu", precision="fp32")
    spec = StudySpec(name="surrey-demo-en", model=model_spec, anchors=anchors, bridges=bridges,
                     include_token_details=True)
    backend = HFBackend(model_spec, resources=spec.resources) if backend_kind == "hf" else ToyBackend()
    if backend_kind == "hf":
        import torch
        torch.manual_seed(42)
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    raw = measure(samples, spec, backend=backend, cache=output/"score-cache")
    save_scores(raw, output/"scores.parquet")
    if raw.frame.status.ne("ok").any():
        raise RuntimeError(f"Native scoring failures: {raw.frame[raw.frame.status.ne('ok')].to_dict('records')}")
    ref = fit_reference(subset(raw, {s.id for s in samples[:6]}))
    save_calibration(ref, output/"reference.json")
    calibrated = transform(raw, ref)
    save_scores(calibrated, output/"calibrated.parquet")
    matrix = to_matrix(calibrated, variant="reference_z")
    save_matrix(matrix, output/"matrix.npz")
    write_json(output/"network.json", semantic_network(matrix, kind="concept"))
    write_json(output/"clusters.json", cluster(matrix, k=2))
    write_json(output/"pca.json", pca(matrix))
    write_json(output/"reliability.json", audit_reliability(raw))
    write_json(output/"template_radii.json", template_radii(calibrated, variant="reference_z"))
    text_matrix = to_matrix(subset(calibrated, {s.id for s in samples[:6]}), variant="reference_z")
    write_json(output/"shift.json", semantic_shift(text_matrix, target_ids=["demo"]*6,
               periods=["early"]*3+["late"]*3, n_bootstrap=20, n_permutations=19))
    replay = measure(samples, spec, backend=backend, cache=output/"score-cache")
    if not np.array_equal(raw.frame.raw_logp.to_numpy(), replay.frame.raw_logp.to_numpy()):
        raise AssertionError("Cached and uncached scores differ.")
    baseline = raw.frame[(raw.frame.sample_id == samples[0].id) & (raw.frame.bridge_id == bridges[0].id)]
    reordered = backend.score_candidates(samples[0], bridges[0], list(reversed(anchors)) + [Anchor(id="extra", surface="cooperation")])
    by_id = {r["anchor_id"]: r for r in reordered}
    errors = {row.anchor_id: abs(row.raw_logp-by_id[row.anchor_id]["raw_logp"])
              for row in baseline.itertuples()}
    if any(errors[row.anchor_id] > 0.01*row.token_count for row in baseline.itertuples()):
        raise AssertionError("Candidate order or prior modality runs changed scores.")
    terminated = backend.score_candidates(samples[0], bridges[0], anchors[:1], event="turn_terminated")[0]
    if terminated["raw_logp"] > by_id[anchors[0].id]["raw_logp"] + 0.01:
        raise AssertionError("Terminated event cannot exceed prefix probability.")
    zh_spec = spec.model_copy(update={"name": "surrey-demo-zh", "anchors": (
        Anchor(id="freedom", surface="自由", language="zh-Hans"),
        Anchor(id="care", surface="关怀", language="zh-Hans"),
        Anchor(id="blue_circle", surface="蓝色圆形", language="zh-Hans")),
        "bridges": (Bridge(id="zh1", prefix="与这份材料相关的一个概念是：\n", language="zh-Hans"),)})
    zh = measure(chinese, zh_spec, backend=backend)
    save_scores(zh, output/"scores-zh.parquet")
    if zh.frame.status.ne("ok").any():
        raise AssertionError("Chinese scoring did not complete.")
    fast_validation = None
    native = None
    if backend_kind == "hf":
        backend.shared_prefill = True
        backend.score_candidates(samples[0], bridges[0], anchors)
        fast_validation = backend.shared_prefill_validation
        if fast_validation is not None:
            fast_validation = {**fast_validation,
                               "enabled": bool(fast_validation["passed"]),
                               "fallback": None if fast_validation["passed"] else "uncached_teacher_forcing"}
        from verify_native import verify_native
        existing_checks = {"modality_state": {"passed": True, "sequence_logp_errors": errors}}
        if fast_validation is not None:
            existing_checks["shared_prefill"] = fast_validation
        native = verify_native(backend, output=output/"native_verification.json",
                               sample=samples[0], bridge=bridges[0],
                               anchors=(anchors[0], anchors[2]), existing_checks=existing_checks)
        if native["status"] != "passed":
            raise AssertionError("Native full-logits reference verification failed; see native_verification.json.")
    report = {"status": "passed", "purpose": "software_demo_not_construct_validation",
              "stimuli": "synthetic_not_human_annotations", "backend": backend_kind,
              "runtime": runtime_manifest(), "hostname": platform.node(),
              "elapsed_seconds": time.perf_counter()-started, "sample_count_en": len(samples),
              "score_items_en": len(raw), "score_items_zh": len(zh),
              "failed_items": 0, "matrix_shape": list(matrix.values.shape),
              "cache_hits_replay": replay.manifest["execution"]["cache_hits"],
              "reordering_logp_errors": errors, "shared_prefill": fast_validation,
              "native_verification": native,
              "model": backend.identity, "future_work": "E1-E6 external validity and full benchmark runs"}
    if backend_kind == "hf":
        report["gpu"] = torch.cuda.get_device_name()
        report["peak_memory_bytes"] = torch.cuda.max_memory_allocated()
        report["cuda_version"] = torch.version.cuda
    write_json(output/"demo_report.json", report)
    print(report, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--backend", choices=["toy", "hf"], default="toy")
    p.add_argument("--skip-video", action="store_true")
    args = p.parse_args()
    try:
        run(args.output.resolve(), args.backend, video=not args.skip_video)
    except Exception as exc:
        write_json(args.output/"demo_report.json", {"status": "failed", "error": str(exc),
                    "traceback": traceback.format_exc(), "runtime": runtime_manifest()})
        raise


if __name__ == "__main__":
    main()
