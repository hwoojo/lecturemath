"""Optional real browser + real V34 E2E. Requires playwright and local Chromium.

Run from repository root: python scripts/verify_web.py /path/to/video.mp4
Uses ports 8000 and 8080. Starts and stops its own servers.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "benchmark-output/web"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    children = []
    errors = []
    with (OUTPUT / "servers.log").open("w") as log:
        try:
            env = {**os.environ, "BOARD_JOBS_DIR": str(OUTPUT / "jobs")}
            children.append(subprocess.Popen([sys.executable,"-m","uvicorn","boardnotes.api:app","--host","127.0.0.1","--port","8000"], cwd=ROOT, env=env, stdout=log, stderr=log))
            children.append(subprocess.Popen([sys.executable,"-m","http.server","8080","--bind","127.0.0.1","--directory","frontend"], cwd=ROOT, stdout=log, stderr=log))
            for _ in range(100):
                try:
                    with urllib.request.urlopen("http://127.0.0.1:8000/health",timeout=2) as response:
                        assert json.load(response)["model_file_present"]
                    break
                except OSError:
                    time.sleep(.1)
            else:
                raise RuntimeError("Server did not start")
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(executable_path="/usr/bin/chromium", headless=True, args=["--no-sandbox"])
                page = browser.new_page(viewport={"width":1100,"height":1000}, accept_downloads=True)
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto("http://127.0.0.1:8080")
                page.wait_for_function("!document.querySelector('#submit').disabled")
                page.screenshot(path=str(OUTPUT / "desktop.png"), full_page=True)
                page.locator("#video").set_input_files(str(Path(sys.argv[1]).resolve()))
                page.locator("#submit").click()
                page.locator("#download").wait_for(state="visible", timeout=180000)
                assert page.locator("#error").is_hidden(), page.locator("#error").inner_text()
                with page.expect_download() as download:
                    page.locator("#download").click()
                download.value.save_as(str(OUTPUT / "browser-result.pdf"))
                page.screenshot(path=str(OUTPUT / "complete.png"), full_page=True)
                page.set_viewport_size({"width":390,"height":844})
                page.screenshot(path=str(OUTPUT / "mobile.png"), full_page=True)
                assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
                assert not errors, errors
                report = {"desktop": "pass", "mobile_no_horizontal_overflow": "pass", "upload_poll_download_real_v34": "pass", "javascript_errors": errors}
                (OUTPUT / "browser-report.json").write_text(json.dumps(report, indent=2))
                print(json.dumps(report), flush=True)
                browser.close()
        finally:
            for child in children:
                child.terminate()
            for child in children:
                child.wait(timeout=15)


if __name__ == "__main__":
    main()
