import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "ACCESS2021_release"


@dataclass
class Settings:
    checkpoint: Path = field(default_factory=lambda: Path(os.getenv("BOARD_MODEL_PATH", str(ROOT / "models/LectureNet_model_BIN_V34_final.dat"))))
    device: str = field(default_factory=lambda: os.getenv("BOARD_DEVICE", "cpu"))
    sampling_fps: float = field(default_factory=lambda: float(os.getenv("BOARD_SAMPLING_FPS", "1")))
    max_width: int = field(default_factory=lambda: int(os.getenv("BOARD_MAX_WIDTH", "1280")))
    max_upload_mb: int = field(default_factory=lambda: int(os.getenv("BOARD_MAX_UPLOAD_MB", "500")))
    max_duration: float = field(default_factory=lambda: float(os.getenv("BOARD_MAX_DURATION", "1800")))
    max_samples: int = field(default_factory=lambda: int(os.getenv("BOARD_MAX_SAMPLES", "1800")))
    max_input_pixels: int = 16777216
    threads: int = field(default_factory=lambda: int(os.getenv("BOARD_THREADS", "2")))
    jobs_dir: Path = field(default_factory=lambda: Path(os.getenv("BOARD_JOBS_DIR", str(ROOT / "jobs"))))
    retention_seconds: int = field(default_factory=lambda: int(os.getenv("BOARD_RETENTION_SECONDS", "86400")))
    keep_intermediates: bool = False
    timestamps: bool = True
    pdf_style: str = field(default_factory=lambda: os.getenv("BOARD_PDF_STYLE", "smooth"))
    config_path: Path = field(default_factory=lambda: Path(os.getenv("BOARD_RESEARCH_CONFIG", str(RELEASE / "configs/FCN_LectureNet.conf"))))
    cors_origins: list[str] = field(default_factory=lambda: os.getenv("BOARD_CORS_ORIGINS", "http://localhost:8080,http://127.0.0.1:8080").split(","))

    def __post_init__(self):
        if self.pdf_style not in ("smooth", "binary"):
            raise ValueError("pdf_style must be smooth or binary")
        if not 0 < self.sampling_fps <= 30:
            raise ValueError("sampling_fps must be in (0, 30]")
        if self.max_width < 64 or min(self.max_upload_mb, self.max_duration, self.max_samples, self.threads, self.retention_seconds) <= 0:
            raise ValueError("Limits must be positive; max_width must be >=64")

    def require_checkpoint(self):
        if not self.checkpoint.is_file():
            raise FileNotFoundError("V34 checkpoint missing. Set BOARD_MODEL_PATH or --model to the pretrained .dat file; no fallback model is used.")
