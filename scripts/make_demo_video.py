"""Generated board-like video for inference smoke tests; not a real lecture."""
import argparse
from pathlib import Path
import cv2
import numpy as np


def create_video(path, seconds=24, fps=10):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (448,256))
    if not writer.isOpened():
        raise RuntimeError("MP4 encoder unavailable")
    for frame_index in range(seconds*fps):
        second = frame_index / fps
        frame = np.full((256,448,3), 245, np.uint8)
        text = "x + y = 12" if second < seconds/2 else "a + b = 24"
        cv2.putText(frame, text, (30,95), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.5, (20,20,20), 2, cv2.LINE_AA)
        cv2.putText(frame, "f(x) = x", (35,180), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.3, (20,20,20), 2, cv2.LINE_AA)
        if 4 <= second < 6:
            # Controlled temporary rectangle occlusion; not a realistic professor.
            cv2.rectangle(frame,(100,30),(230,130),(140,110,85),-1)
        writer.write(frame)
    writer.release()
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--seconds", type=int, default=24)
    args = parser.parse_args()
    print(create_video(args.output, args.seconds))
