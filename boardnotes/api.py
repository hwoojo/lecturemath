import asyncio
import contextlib
import json
import logging
import shutil
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict, replace
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from .config import ROOT, Settings
from .video import metadata

logger = logging.getLogger(__name__)


def run_subprocess(jobdir, settings):
    (jobdir / "settings.json").write_text(json.dumps(asdict(settings), default=str))
    with (jobdir / "worker.log").open("w") as log:
        result = subprocess.run([sys.executable, "-m", "boardnotes.worker", str(jobdir)], cwd=ROOT,
                                stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        error_file = jobdir / "error.json"
        message = json.loads(error_file.read_text())["error"] if error_file.exists() else "Worker exited unexpectedly; consult the server job log."
        raise RuntimeError(message)


class Jobs:
    def __init__(self, settings, runner):
        self.settings, self.runner = settings, runner
        self.root = settings.jobs_dir.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.busy = False
        self.records = {}
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")
        for path in self.root.glob("*/status.json"):
            record = json.loads(path.read_text())
            if record["status"] in ("queued", "running", "uploading"):
                record.update(status="failed", error="Server restarted before this job completed", finished_at=time.time())
                self.clean_work(path.parent)
            self.records[path.parent.name] = record
            self.write(path.parent.name)
        # Incomplete uploads may not have a status file after a server crash.
        for path in self.root.iterdir():
            if path.is_dir() and len(path.name) == 32 and path.name not in self.records:
                shutil.rmtree(path)
        self.cleanup()

    def write(self, job_id):
        path = self.root / job_id
        (path / "status.tmp").write_text(json.dumps(self.records[job_id]))
        (path / "status.tmp").replace(path / "status.json")

    def clean_work(self, path):
        (path / "input.video").unlink(missing_ok=True)
        (path / "settings.json").unlink(missing_ok=True)
        if not self.settings.keep_intermediates:
            for name in ("temporal", "masks", "pages"):
                shutil.rmtree(path / name, ignore_errors=True)

    def cleanup(self):
        with self.lock:
            now = time.time()
            for job_id, record in list(self.records.items()):
                if record["status"] in ("complete", "failed") and now - record["finished_at"] > self.settings.retention_seconds:
                    shutil.rmtree(self.root / job_id, ignore_errors=True)
                    del self.records[job_id]

    def execute(self, job_id, settings):
        path = self.root / job_id
        with self.lock:
            self.records[job_id].update(status="running", stage="loading_model", progress=1)
            self.write(job_id)
        try:
            self.runner(path, settings)
            if not (path / "result.pdf").is_file():
                raise RuntimeError("Worker did not produce a PDF")
            with self.lock:
                self.records[job_id].update(status="complete", stage="complete", progress=100)
        except Exception as error:
            logger.exception("Job %s failed", job_id)
            with self.lock:
                self.records[job_id].update(status="failed", stage="failed", error=str(error))
            (path / "result.pdf").unlink(missing_ok=True)
        finally:
            self.clean_work(path)
            with self.lock:
                self.records[job_id]["finished_at"] = time.time()
                self.write(job_id)
                self.busy = False


def create_app(settings=None, runner=run_subprocess):
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        manager = Jobs(settings, runner)
        app.state.jobs = manager

        async def janitor():
            while True:
                await asyncio.sleep(min(60, settings.retention_seconds))
                await asyncio.to_thread(manager.cleanup)

        task = asyncio.create_task(janitor())
        yield
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await asyncio.to_thread(manager.pool.shutdown, wait=True)

    app = FastAPI(title="Boardnotes V1", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type"],
                       expose_headers=["Content-Disposition"])

    @app.get("/health")
    def health():
        return {"status": "ok", "model_file_present": settings.checkpoint.is_file(),
                "device": settings.device, "max_upload_mb": settings.max_upload_mb,
                "max_duration_seconds": settings.max_duration}

    @app.post("/jobs", status_code=202)
    async def upload(request: Request, fps: float | None = Query(None, gt=0, le=30), timestamps: bool = True):
        """Raw video body, streamed to disk; avoids multipart buffering before size checks."""
        try:
            settings.require_checkpoint()
        except FileNotFoundError as error:
            raise HTTPException(503, str(error)) from error
        manager = app.state.jobs
        with manager.lock:
            if manager.busy:
                raise HTTPException(409, "Another upload or inference job is active. Try again after it finishes.")
            manager.busy = True
        job_id = uuid.uuid4().hex
        path = manager.root / job_id
        submitted = False
        try:
            path.mkdir()
            limit = settings.max_upload_mb * 1024*1024
            length = request.headers.get("content-length")
            if length:
                try:
                    if int(length) > limit:
                        raise HTTPException(413, "Upload exceeds size limit")
                except ValueError as error:
                    raise HTTPException(400, "Invalid Content-Length") from error
            size = 0
            with (path / "input.video").open("wb") as stream:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(413, "Upload exceeds size limit")
                    await asyncio.to_thread(stream.write, chunk)
            if not size:
                raise HTTPException(400, "Empty video upload")
            job_settings = replace(settings, sampling_fps=fps or settings.sampling_fps, timestamps=timestamps)
            try:
                meta = await asyncio.to_thread(metadata, path / "input.video", job_settings)
            except ValueError as error:
                raise HTTPException(422, str(error)) from error
            with manager.lock:
                manager.records[job_id] = {"id": job_id, "status": "queued", "stage": "queued", "progress": 0,
                                           "created_at": time.time(), "metadata": meta, "error": None}
                manager.write(job_id)
            manager.pool.submit(manager.execute, job_id, job_settings)
            submitted = True
            return {"id": job_id, "status": "queued", "status_url": f"/jobs/{job_id}"}
        finally:
            if not submitted:
                shutil.rmtree(path, ignore_errors=True)
                with manager.lock:
                    manager.records.pop(job_id, None)
                    manager.busy = False

    def get_record(job_id):
        manager = app.state.jobs
        with manager.lock:
            if job_id not in manager.records:
                raise HTTPException(404, "Job not found or expired")
            return dict(manager.records[job_id])

    @app.get("/jobs/{job_id}")
    def status(job_id: str):
        record = get_record(job_id)
        progress_file = app.state.jobs.root / job_id / "progress.json"
        if record["status"] == "running" and progress_file.exists():
            record.update(json.loads(progress_file.read_text()))
        if record["status"] == "complete":
            record["download_url"] = f"/jobs/{job_id}/pdf"
        return record

    @app.get("/jobs/{job_id}/pdf")
    def download(job_id: str):
        record = get_record(job_id)
        if record["status"] != "complete":
            raise HTTPException(409, "PDF is not ready")
        return FileResponse(app.state.jobs.root / job_id / "result.pdf", media_type="application/pdf", filename="lecture-notes.pdf")

    return app


app = create_app()
