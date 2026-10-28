import argparse
import json
import logging
import sys
from pathlib import Path
from .config import Settings
from .pipeline import process_video


def main():
    parser = argparse.ArgumentParser(description="FCN-LectureNet + ST3D video to board PDF")
    parser.add_argument("video", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--fps", type=float)
    parser.add_argument("--max-width", type=int)
    parser.add_argument("--research-config", type=Path)
    parser.add_argument("--pdf-style", choices=("smooth", "binary"))
    parser.add_argument("--keep-intermediates", action="store_true")
    parser.add_argument("--export-v2-inputs", action="store_true", help="Export uncropped ST3D keyframes and coordinate/time manifest")
    parser.add_argument("--no-timestamps", action="store_true")
    args = parser.parse_args()
    overrides = {key: value for key, value in {"checkpoint": args.model, "device": args.device,
                 "sampling_fps": args.fps, "max_width": args.max_width, "config_path": args.research_config,
                 "pdf_style": args.pdf_style}.items() if value is not None}
    settings = Settings(**overrides, keep_intermediates=args.keep_intermediates, export_v2_inputs=args.export_v2_inputs, timestamps=not args.no_timestamps)
    logging.basicConfig(level=logging.INFO)
    try:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise ValueError("Output directory must be empty (prevents overwriting prior results)")
        if args.video.stat().st_size > settings.max_upload_mb * 1024*1024:
            raise ValueError("Video exceeds upload size limit")
        report = process_video(args.video, args.output_dir, settings, lambda stage, pct: print(f"{pct:3}% {stage}", file=sys.stderr, flush=True))
    except Exception as error:
        logging.error("Processing failed: %s", error)
        raise SystemExit(1) from error
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
