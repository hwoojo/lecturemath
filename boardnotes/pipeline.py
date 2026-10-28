import contextlib
import hashlib
import json
import logging
import resource
import time
from pathlib import Path
import cv2
from .model import LectureNet
from .video import metadata, frames
from .temporal import summarize
from .pdf import generate_pdf

logger = logging.getLogger(__name__)


def process_video(video, workdir, settings, progress=lambda *args: None):
    started = time.monotonic()
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    meta = metadata(video, settings)
    progress("loading_model", 2)
    model = LectureNet(settings)
    times, indices, compressed = [], [], []
    with (workdir / "pipeline.log").open("w") as log, contextlib.redirect_stdout(log):
        for idx, milliseconds, frame in frames(video, meta, settings):
            binary = model.extract(frame)
            ok, png = cv2.imencode(".png", binary)
            if not ok:
                raise RuntimeError("Failed to encode extracted mask")
            compressed.append(png)
            indices.append(idx)
            times.append(milliseconds)
            if settings.keep_intermediates:
                folder = workdir / "masks"
                folder.mkdir(exist_ok=True)
                (folder / f"{idx:08}.png").write_bytes(png.tobytes())
            progress("binarizing", min(58, 5 + int(53 * len(times) / max(1, meta["estimated_samples"]))))
        del model
        if not times:
            raise ValueError("No frames could be sampled")
        keyframes, stats = summarize((times, indices, compressed), settings, workdir / "temporal", progress)
        progress("pdf", 96)
        count = generate_pdf(keyframes, workdir / "result.pdf", settings.timestamps, workdir / "pages" if settings.keep_intermediates else None, style=settings.pdf_style)
    digest = hashlib.sha256()
    with settings.checkpoint.open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024*1024), b""):
            digest.update(chunk)
    report = {"model": "FCN-LectureNet (user-provided V34 state_dict)",
              "checkpoint_sha256": digest.hexdigest(), "device": settings.device,
              "metadata": meta, "sampling_fps": settings.sampling_fps,
              "max_width": settings.max_width, "pdf_style": settings.pdf_style, "pages": count, **stats,
              "elapsed_seconds": round(time.monotonic() - started, 3),
              "process_peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2)}
    (workdir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Completed: pages=%s elapsed=%ss", count, report["elapsed_seconds"])
    return report
