import math
import cv2


def metadata(path, settings):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("Cannot decode video")
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not all(math.isfinite(x) and x > 0 for x in (fps, count, width, height)):
            raise ValueError("Video has invalid duration, frame count, FPS or dimensions")
        duration = count / fps
        if duration > settings.max_duration:
            raise ValueError(f"Video exceeds duration limit ({settings.max_duration}s)")
        if width * height > settings.max_input_pixels:
            raise ValueError("Video resolution exceeds configured pixel limit")
        samples = math.ceil(duration * min(settings.sampling_fps, fps))
        if samples > settings.max_samples:
            raise ValueError(f"Video exceeds sample limit ({settings.max_samples})")
        return dict(fps=fps, frame_count=int(count), width=width, height=height, duration=duration, estimated_samples=samples)
    finally:
        cap.release()


def frames(path, meta, settings):
    """Sequential decode avoids unreliable repeated seeks and duplicate samples."""
    cap = cv2.VideoCapture(str(path))
    target, idx, emitted = 0.0, 0, 0
    step = max(1.0, meta["fps"] / settings.sampling_fps)
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            seconds = idx / meta["fps"]
            if seconds >= settings.max_duration or idx >= meta["frame_count"]:
                raise ValueError("Decoded video exceeds declared or configured duration")
            if idx >= round(target):
                ok, frame = cap.retrieve()
                if not ok:
                    raise ValueError("Video frame decoding failed")
                if emitted >= settings.max_samples:
                    raise ValueError("Decoded video exceeds sample limit")
                if frame.shape[1] > settings.max_width:
                    scale = settings.max_width / frame.shape[1]
                    frame = cv2.resize(frame, (settings.max_width, max(32, round(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA)
                if min(frame.shape[:2]) < 32:
                    raise ValueError("Frame dimensions must be at least 32 pixels")
                yield idx, seconds * 1000, frame
                emitted += 1
                target += step
            idx += 1
        if idx < meta["frame_count"] - max(2, meta["fps"]):
            raise ValueError("Video is truncated or failed to decode fully")
    finally:
        cap.release()
