import json
import threading
import time
import pytest
from fastapi.testclient import TestClient
from boardnotes.api import create_app, Jobs
from boardnotes.config import Settings
from boardnotes.pdf import generate_pdf
from test_video_model import make_video
import numpy as np


def settings(tmp_path, **overrides):
    # This sentinel enables API tests only; fake runner never loads it as a model.
    checkpoint = tmp_path / "sentinel.dat"
    checkpoint.write_bytes(b"NOT A MODEL")
    return Settings(checkpoint=checkpoint, jobs_dir=tmp_path / "jobs", **overrides)


def fake_runner(path, _):
    image = np.full((64,128),255,np.uint8)
    image[20:40,20:100] = 0
    generate_pdf([(image,0,1)], path / "result.pdf")


def wait_job(client, job_id):
    deadline = time.monotonic()+10
    while time.monotonic() < deadline:
        result = client.get(f"/jobs/{job_id}").json()
        if result["status"] in ("complete", "failed"):
            return result
        time.sleep(.02)
    pytest.fail("Job did not finish")


def test_upload_status_download_cleanup(tmp_path):
    config = settings(tmp_path)
    video = make_video(tmp_path / "sample.mp4")
    with TestClient(create_app(config, fake_runner)) as client:
        response = client.post("/jobs?fps=1&timestamps=false", content=video.read_bytes())
        assert response.status_code == 202, response.text
        job_id = response.json()["id"]
        job = wait_job(client, job_id)
        assert job["status"] == "complete", job
        pdf = client.get(job["download_url"])
        assert pdf.headers["content-type"] == "application/pdf" and pdf.content.startswith(b"%PDF")
        assert not (config.jobs_dir / job_id / "input.video").exists()
        assert client.get("/jobs/not-a-job").status_code == 404
        assert client.get("/jobs/..%2fanything/pdf").status_code == 404
        preflight = client.options("/jobs", headers={"Origin":"http://localhost:8080", "Access-Control-Request-Method":"POST", "Access-Control-Request-Headers":"content-type"})
        assert preflight.headers["access-control-allow-origin"] == "http://localhost:8080"


def test_missing_model_invalid_upload_and_size(tmp_path):
    with TestClient(create_app(Settings(checkpoint=tmp_path / "missing", jobs_dir=tmp_path / "missing-jobs"))) as client:
        assert client.post("/jobs", content=b"anything").status_code == 503
    config = settings(tmp_path, max_upload_mb=1)
    with TestClient(create_app(config, fake_runner)) as client:
        assert client.post("/jobs", content=b"").status_code == 400
        assert client.post("/jobs", content=b"broken").status_code == 422
        assert client.post("/jobs", content=b"x"*(1024*1024+1)).status_code == 413
        assert client.post("/jobs", content=iter([b"x"*600000,b"x"*600000])).status_code == 413
        assert not list(config.jobs_dir.iterdir())


def test_busy_and_error_state(tmp_path):
    started, release = threading.Event(), threading.Event()
    def blocked_runner(path, _):
        started.set()
        release.wait(5)
        raise ValueError("controlled test failure")
    config = settings(tmp_path)
    video = make_video(tmp_path / "sample.mp4")
    with TestClient(create_app(config, blocked_runner)) as client:
        response = client.post("/jobs", content=video.read_bytes())
        job_id = response.json()["id"]
        assert started.wait(2)
        assert client.post("/jobs", content=video.read_bytes()).status_code == 409
        assert client.get(f"/jobs/{job_id}/pdf").status_code == 409
        release.set()
        assert wait_job(client, job_id)["status"] == "failed"
        assert not (config.jobs_dir / job_id / "input.video").exists()


def test_restart_and_expiration(tmp_path):
    config = settings(tmp_path, retention_seconds=1)
    path = config.jobs_dir / ("a"*32)
    path.mkdir(parents=True)
    (path / "status.json").write_text(json.dumps({"id":path.name,"status":"running","created_at":time.time()}))
    (path / "input.video").write_bytes(b"partial")
    manager = Jobs(config, fake_runner)
    assert manager.records[path.name]["status"] == "failed"
    assert not (path / "input.video").exists()
    manager.records[path.name]["finished_at"] = time.time()-2
    manager.cleanup()
    assert not path.exists() and not manager.records
    manager.pool.shutdown()
