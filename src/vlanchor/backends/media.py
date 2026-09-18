"""Freeze local media once, retaining the evidence needed to reproduce scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from ..errors import MissingMedia, ResourceUnavailable, VLanchorError
from ..types import Part, ResourceProfile, Sample


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class FrozenMedia:
    content: list[dict[str, Any]] = field(default_factory=list)
    images: list[Image.Image] = field(default_factory=list)
    videos: list[np.ndarray] = field(default_factory=list)
    video_metadata: list[dict[str, Any]] = field(default_factory=list)
    records: list[dict[str, Any]] = field(default_factory=list)
    fingerprint: str = ""


def _path(part: Part) -> Path:
    path = Path(part.path or "").expanduser().resolve()
    if not path.is_file():
        raise MissingMedia(f"Media file does not exist: {path}")
    return path


def _load_image(path: Path) -> tuple[Image.Image, dict[str, Any]]:
    try:
        with Image.open(path) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB").copy()
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise MissingMedia(f"Cannot decode image {path}: {exc}") from exc
    record = {
        "type": "image", "path": str(path), "width": rgb.width, "height": rgb.height,
        "rgb_sha256": _digest(rgb.tobytes()), "orientation": "exif_transposed",
    }
    return rgb, record


def _load_video(
    path: Path, part: Part, resources: ResourceProfile,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    try:
        import av
    except ImportError as exc:
        raise ResourceUnavailable("Video decoding requires PyAV: install vlanchor[hf].") from exc

    # First pass retains timestamps only, not all decoded RGB frames in RAM.
    timestamps: list[float] = []
    inferred_timestamps = False
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise MissingMedia(f"No video stream in {path}")
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate else 0.0
            if not np.isfinite(fps) or fps <= 0:
                raise MissingMedia(f"Video needs a valid source frame rate: {path}")
            origin: float | None = None
            for index, frame in enumerate(container.decode(stream)):
                if frame.time is None:
                    inferred_timestamps = True
                    timestamp = index / fps
                else:
                    if origin is None:
                        origin = float(frame.time)
                    timestamp = float(frame.time) - origin
                timestamps.append(timestamp)
        if not timestamps:
            raise MissingMedia(f"Video contains no decoded frames: {path}")
        start = part.clip_start or 0.0
        end = part.clip_end if part.clip_end is not None else float("inf")
        eligible = np.flatnonzero((np.asarray(timestamps) >= start)
                                  & (np.asarray(timestamps) < end))
        if eligible.size == 0:
            raise MissingMedia(f"Clip contains no frames: {path} [{start}, {end})")
        positions = np.linspace(0, eligible.size - 1, resources.video.sampled_frames)
        indices = eligible[np.rint(positions).astype(int)].tolist()
        wanted = set(indices)
        decoded: dict[int, np.ndarray] = {}
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            for index, frame in enumerate(container.decode(stream)):
                if index in wanted:
                    decoded[index] = frame.to_ndarray(format="rgb24")
                if index >= max(wanted):
                    break
        if set(decoded) != wanted:
            raise MissingMedia(f"Video changed or failed during second decoding pass: {path}")
        frames = np.stack([decoded[index] for index in indices])
    except VLanchorError:
        raise
    except Exception as exc:
        raise MissingMedia(f"Cannot decode video {path}: {exc}") from exc

    selected_times = [timestamps[i] for i in indices]
    # Qwen's native timestamp renderer uses frame_index / fps. Detect recordings
    # where that would misrepresent time rather than silently treating them as CFR.
    errors = np.abs(np.asarray(timestamps) - np.arange(len(timestamps)) / fps)
    constant_rate = bool(errors.max() <= 0.25 / fps + 1e-6)
    metadata = {
        "fps": fps, "total_num_frames": len(timestamps),
        "duration": timestamps[-1] + 1.0 / fps, "frames_indices": indices,
    }
    record = {
        "type": "video", "path": str(path), **metadata,
        "sampled_frames": len(indices), "unique_frame_count": len(wanted),
        "timestamps_seconds": selected_times, "constant_frame_rate": constant_rate,
        "timestamps_inferred": inferred_timestamps,
        "clip_start": part.clip_start, "clip_end": part.clip_end,
        "height": int(frames.shape[1]), "width": int(frames.shape[2]),
        "frame_rgb_sha256": [_digest(frame.tobytes()) for frame in frames],
    }
    frames.setflags(write=False)
    return frames, metadata, record


def freeze_media(sample: Sample, resources: ResourceProfile) -> FrozenMedia:
    """Preserve input order and exact text; never caption, crop, or re-sample per anchor."""
    frozen = FrozenMedia()
    fingerprint_parts: list[dict[str, Any]] = []
    for part in sample.parts:
        if part.type == "text":
            item = {"type": "text", "text": part.text}
            frozen.content.append(item)
            fingerprint_parts.append(item)
        elif part.type == "image":
            path = _path(part)
            image, record = _load_image(path)
            frozen.images.append(image)
            frozen.content.append({"type": "image", "image": str(path)})
            frozen.records.append(record)
            fingerprint_parts.append({k: v for k, v in record.items() if k != "path"})
        else:
            path = _path(part)
            frames, metadata, record = _load_video(path, part, resources)
            frozen.videos.append(frames)
            frozen.video_metadata.append(metadata)
            frozen.content.append({"type": "video", "video": str(path)})
            frozen.records.append(record)
            fingerprint_parts.append({k: v for k, v in record.items() if k != "path"})
    frozen.fingerprint = _digest(json.dumps(
        fingerprint_parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8"))
    return frozen


def processor_media_kwargs(
    frozen: FrozenMedia, resources: ResourceProfile, *, temporal_patch_size: int = 2,
    cap_pixels_supported: bool = False,
) -> dict[str, Any]:
    """Map public pixel budgets to native image-area and video-total-area budgets."""
    kwargs: dict[str, Any] = {}
    if frozen.images:
        kwargs["images"] = frozen.images
        kwargs["images_kwargs"] = {"do_resize": True, "size": {
            "shortest_edge": resources.image.min_pixels,
            "longest_edge": resources.image.max_pixels,
        }}
    if frozen.videos:
        if any(not r["constant_frame_rate"] for r in frozen.records if r["type"] == "video"):
            raise VLanchorError(
                "Variable-frame-rate timestamps cannot be faithfully rendered by this native "
                "Qwen adapter. Convert explicitly to constant frame rate and record that transform."
            )
        frames = resources.video.sampled_frames
        if frames < temporal_patch_size:
            raise VLanchorError("sampled_frames is smaller than the native temporal patch size.")
        kwargs["videos"] = [video.copy() for video in frozen.videos]
        video_kwargs: dict[str, Any] = {
            "do_resize": True, "do_sample_frames": False, "return_metadata": True,
            "video_metadata": [dict(m, frames_indices=list(m["frames_indices"]))
                               for m in frozen.video_metadata],
            "size": {"shortest_edge": 4096,
                     # Native resize computes its scale using raw frame count,
                     # even when temporal padding will duplicate the last frame.
                     "longest_edge": frames * resources.video.max_pixels_per_frame},
        }
        if cap_pixels_supported:
            video_kwargs["cap_pixels_per_frame"] = False
        kwargs["videos_kwargs"] = video_kwargs
    return kwargs
