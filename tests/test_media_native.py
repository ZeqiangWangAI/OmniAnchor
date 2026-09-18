from fractions import Fraction

import numpy as np
from PIL import Image
import pytest

from omnianchor.backends.media import FrozenMedia, freeze_media, processor_media_kwargs
from omnianchor.errors import MissingMedia, OmniAnchorError
from omnianchor.types import Part, ResourceProfile, Sample, VideoBudget


def test_image_orientation_order_and_path_independent_pixel_fingerprint(tmp_path):
    path = tmp_path / "oriented.jpg"
    image = Image.new("RGB", (10, 20), "red")
    exif = image.getexif()
    exif[274] = 6
    image.save(path, exif=exif)
    parts = (Part(type="text", text="before"), Part(type="image", path=str(path)),
             Part(type="text", text="after"))
    frozen = freeze_media(Sample(id="s", parts=parts), ResourceProfile())
    assert [part["type"] for part in frozen.content] == ["text", "image", "text"]
    assert frozen.images[0].size == (20, 10)
    copy = tmp_path / "copied.jpg"
    copy.write_bytes(path.read_bytes())
    changed_parts = (parts[0], Part(type="image", path=str(copy)), parts[2])
    copied = freeze_media(Sample(id="other", parts=changed_parts), ResourceProfile())
    assert copied.fingerprint == frozen.fingerprint


def test_missing_or_corrupt_media_raises_explicit_error(tmp_path):
    for path in (tmp_path / "missing.png", tmp_path / "bad.png"):
        if path.name == "bad.png":
            path.write_bytes(b"not an image")
        with pytest.raises(MissingMedia):
            freeze_media(Sample(id="s", parts=(Part(type="image", path=str(path)),)),
                         ResourceProfile())


def frozen_video(frames=8, constant_rate=True):
    return FrozenMedia(videos=[np.zeros((frames, 32, 48, 3), dtype=np.uint8)],
                       video_metadata=[{"fps": 29.97, "frames_indices": list(range(frames)),
                                        "total_num_frames": frames, "duration": frames / 29.97}],
                       records=[{"type": "video", "constant_frame_rate": constant_rate}])


def test_native_pixel_profiles_use_area_and_total_video_pixels_not_guessed_tokens():
    frozen = frozen_video()
    frozen.images = [Image.new("RGB", (20, 30))]
    kwargs = processor_media_kwargs(frozen, ResourceProfile(), cap_pixels_supported=True)
    assert kwargs["images_kwargs"]["size"] == {"shortest_edge": 65536, "longest_edge": 262144}
    video = kwargs["videos_kwargs"]
    assert video["size"] == {"shortest_edge": 4096, "longest_edge": 8 * 65536}
    assert video["do_sample_frames"] is False and video["cap_pixels_per_frame"] is False
    assert video["video_metadata"][0]["fps"] == 29.97
    # Native processing cannot mutate the frozen recording or frame indices.
    kwargs["videos"][0][0, 0, 0, 0] = 255
    video["video_metadata"][0]["frames_indices"][0] = 100
    assert frozen.videos[0][0, 0, 0, 0] == 0
    assert frozen.video_metadata[0]["frames_indices"][0] == 0


def test_temporal_padding_budget_and_vfr_fail_closed():
    resources = ResourceProfile(video=VideoBudget(sampled_frames=7))
    kwargs = processor_media_kwargs(frozen_video(7), resources)
    assert kwargs["videos_kwargs"]["size"]["longest_edge"] == 7 * 65536
    assert "cap_pixels_per_frame" not in kwargs["videos_kwargs"]
    with pytest.raises(OmniAnchorError, match="Variable-frame-rate"):
        processor_media_kwargs(frozen_video(constant_rate=False), ResourceProfile())


def test_pyav_freezes_true_fps_clip_indices_and_decoded_pixels(tmp_path):
    av = pytest.importorskip("av")
    path = tmp_path / "tiny.mp4"
    fps = Fraction(30000, 1001)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for index in range(12):
            rgb = np.full((32, 32, 3), index * 20, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    sample = Sample(id="v", parts=(Part(type="video", path=str(path),
                                        clip_start=0.10, clip_end=0.30),))
    frozen = freeze_media(sample, ResourceProfile())
    record = frozen.records[0]
    assert record["fps"] == pytest.approx(float(fps))
    assert record["sampled_frames"] == 8
    assert all(0.10 <= t < 0.30 for t in record["timestamps_seconds"])
    assert record["constant_frame_rate"] is True
    assert record["unique_frame_count"] < 8  # Short clips preserve repeated indices explicitly.
    assert frozen.videos[0].shape == (8, 32, 32, 3)
    assert not frozen.videos[0].flags.writeable
    assert frozen.video_metadata[0]["frames_indices"] == record["frames_indices"]
    assert freeze_media(sample, ResourceProfile()).fingerprint == frozen.fingerprint
