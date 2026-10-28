import cv2
import numpy as np
import pytest
from boardnotes.config import Settings
from boardnotes.model import LectureNet
from boardnotes.video import metadata, frames


def make_video(path, seconds=2, fps=10):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 96))
    assert writer.isOpened()
    for index in range(seconds*fps):
        image = np.full((96,160,3), 255, np.uint8)
        cv2.putText(image, str(index), (15,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        writer.write(image)
    writer.release()
    return path


def test_metadata_sampling_and_limits(tmp_path):
    video = make_video(tmp_path / "input.mp4")
    settings = Settings(max_width=80)
    meta = metadata(video, settings)
    sampled = list(frames(video, meta, settings))
    assert meta["duration"] == 2 and meta["width"] == 160
    assert [item[0] for item in sampled] == [0,10]
    assert [item[1] for item in sampled] == [0,1000]
    assert sampled[0][2].shape == (48,80,3)
    with pytest.raises(ValueError, match="duration"):
        metadata(video, Settings(max_duration=1))
    with pytest.raises(ValueError, match="sample"):
        metadata(video, Settings(max_samples=1))
    (tmp_path / "bad").write_bytes(b"not video")
    with pytest.raises(ValueError, match="decode"):
        metadata(tmp_path / "bad", settings)


def test_no_checkpoint_never_uses_fallback(tmp_path):
    with pytest.raises(FileNotFoundError, match="no fallback"):
        LectureNet(Settings(checkpoint=tmp_path / "missing.dat"))


@pytest.mark.parametrize("fps", [0,-1,31])
def test_invalid_sample_fps(fps):
    with pytest.raises(ValueError):
        Settings(sampling_fps=fps)
