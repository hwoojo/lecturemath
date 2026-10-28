"""One subprocess per job releases inference memory on exit."""
import json
import logging
import sys
from pathlib import Path
from .config import Settings
from .pipeline import process_video


def main():
    jobdir = Path(sys.argv[1])
    values = json.loads((jobdir / "settings.json").read_text())
    for key in ("checkpoint", "jobs_dir", "config_path"):
        values[key] = Path(values[key])
    logging.basicConfig(level=logging.INFO)

    def progress(stage, percent):
        temporary = jobdir / "progress.tmp"
        temporary.write_text(json.dumps({"stage": stage, "progress": percent}))
        temporary.replace(jobdir / "progress.json")

    try:
        process_video(jobdir / "input.video", jobdir, Settings(**values), progress)
    except Exception as error:
        logging.exception("Job failed")
        # Server paths and traceback remain in server logs, not public API responses.
        message = str(error) if isinstance(error, ValueError) else "Processing failed; consult the server job log."
        (jobdir / "error.json").write_text(json.dumps({"error": message}))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
