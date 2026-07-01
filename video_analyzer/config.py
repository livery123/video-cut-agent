from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT_DIR / ".env")
    except ImportError:
        pass


_load_dotenv()

# ── 路径 ──────────────────────────────────────────────
UPLOADS_DIR = ROOT_DIR / "uploads"
TASKS_DIR = ROOT_DIR / "tasks"
DEBUG_DIR = ROOT_DIR / "debug"
EVAL_DATASET_DIR = ROOT_DIR / "eval" / "dataset"

for _d in (UPLOADS_DIR, TASKS_DIR, DEBUG_DIR, EVAL_DATASET_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── 抽帧 ──────────────────────────────────────────────
COARSE_SAMPLE_INTERVAL = 12
MATCH_FRAME_WIDTH = 480

# ── OCR 并行 ──────────────────────────────────────────
OCR_WORKERS = 8

# ── 局开始：ROI 内识别「你已选择」 ─────────────────────
ROUND_START_KEYWORD = "你已选择"
ROUND_START_OCR_THRESHOLD = 80
ROUND_START_ROI = (0.22, 0.26, 0.78, 0.36)
MIN_GAP_BETWEEN_ROUNDS = 120

# ── 任务限制 ─────────────────────────────────────────
MAX_FILE_SIZE_MB = 2048
MAX_DURATION_SECONDS = 7200
SUPPORTED_FORMATS = ["mp4", "mkv", "flv", "webm", "avi"]
