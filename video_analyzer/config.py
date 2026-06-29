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
TEMPLATES_DIR = ROOT_DIR / "templates"
UPLOADS_DIR = ROOT_DIR / "uploads"
TASKS_DIR = ROOT_DIR / "tasks"
DEBUG_DIR = ROOT_DIR / "debug"
EVAL_DATASET_DIR = ROOT_DIR / "eval" / "dataset"

for _d in (TEMPLATES_DIR, UPLOADS_DIR, TASKS_DIR, DEBUG_DIR, EVAL_DATASET_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── 抽帧 ──────────────────────────────────────────────
COARSE_SAMPLE_INTERVAL = 2
REFINE_SAMPLE_INTERVAL = 0.5
REFINE_WINDOW = 15
FRAME_WIDTH = 960
# 模板匹配专用：缩小帧宽 + 预计算尺度，显著提速
MATCH_FRAME_WIDTH = 480
MATCH_SCALES = (0.5, 0.6, 0.75, 0.85, 1.0, 1.15)
MATCH_WORKERS = 0

# ── 帧去重 ────────────────────────────────────────────
FRAME_SIMILARITY_THRESHOLD = 0.95

# ── 单帧状态 ─────────────────────────────────────────
MIN_STATE_SCORE = 3
STATES = ("LOBBY", "ROLE", "PLAYING", "ENDING", "UNKNOWN")

MIN_STATE_DURATION = {
    "LOBBY": 4,
    "ROLE": 2,
    "PLAYING": 8,
    "ENDING": 2,
}

MAX_ISOLATED_FRAMES = 2

# ── 分局规则 ─────────────────────────────────────────
START_OFFSET = 2
END_OFFSET = 2
ENDING_CONFIRM_FRAMES = 2
ROUND_COOLDOWN = 30
# ── 分局：ROI 内 OCR 识别「你已选择」 ─────────────────
ROUND_START_KEYWORD = "你已选择"
ROUND_START_OCR_THRESHOLD = 80          # 模糊匹配 0–100
# 「你已选择！」标题带（496×368 录屏实测 y≈26%–36%）
ROUND_START_ROI = (0.22, 0.26, 0.78, 0.36)

MIN_GAP_BETWEEN_ROUNDS = 120       # 两局开始至少间隔 2 分钟
MIN_ROUND_DURATION = 120
MAX_ROUND_DURATION = 2100

# ── 置信度 ───────────────────────────────────────────
CONF_WEIGHTS = {
    "start": 0.35,
    "end": 0.35,
    "duration": 0.15,
    "continuity": 0.15,
}

START_CONFIDENCE = {
    ("LOBBY", "ROLE"): 0.96,
    ("UNKNOWN", "ROLE"): 0.93,
    ("ENDING", "ROLE"): 0.90,
    ("LOBBY", "PLAYING"): 0.80,
    ("UNKNOWN", "PLAYING"): 0.65,
}

END_CONFIDENCE = {
    "ending_to_lobby": 0.92,
    "ending_streak": 0.90,               # 无「返回大厅」按钮时主要依赖结算界面
    "playing_to_lobby": 0.78,
    "video_truncated": 0.60,
}

# ── OCR ───────────────────────────────────────────────
FUZZY_MATCH_THRESHOLD = 80
DARK_RATIO_THRESHOLD = 0.7

ROI_REGIONS = {
    "bottom": (0.0, 0.80, 1.0, 1.0),
    "center": (0.0, 0.30, 1.0, 0.70),
    "role_center": (0.28, 0.18, 0.72, 0.58),  # 「你已选择」+ 角色名 + 角色图标（任意颜色）
    "role_left": (0.0, 0.10, 0.22, 0.90),     # 左侧玩家列表 + 绿色勾
    "top": (0.0, 0.0, 1.0, 0.15),
}

KEYWORDS = {
    "LOBBY": ["开始游戏", "准备", "房间", "房间码", "邀请", "玩家"],
    "ROLE": [
        "你已选择",
        "你的身份",
        "你是",
        "你可以",
        "随机",
        "号",
        "鹅",
        "鸭",
        "中立",
    ],
    "ENDING": ["胜利", "失败", "鹅胜利", "鸭子胜利", "中立胜利", "继续"],
    "PLAYING": ["任务", "投票", "跳过", "会议", "讨论"],
}

OCR_SCORE_RULES = [
    (("开始游戏", "准备"), "LOBBY", 4),
    (("你已选择",), "ROLE", 8),           # 角色揭示主锚点
    (("你的身份", "你是"), "ROLE", 5),
    (("你可以",), "ROLE", 4),             # 「你可以隐身。」等角色描述
    (("胜利", "失败"), "ENDING", 5),
    (("投票", "跳过"), "PLAYING", 2),
]

# ── 模板匹配（V1 实际只用 role_chosen_text，其余保留兼容） ──
TEMPLATE_SCORES = {
    "role_chosen_text.png": [("ROLE", 8)],
}

TEMPLATE_MATCH_THRESHOLD = 0.75

# ── 任务限制 ─────────────────────────────────────────
MAX_FILE_SIZE_MB = 2048
MAX_DURATION_SECONDS = 7200
SUPPORTED_FORMATS = ["mp4", "mkv", "flv", "webm", "avi"]

# ── 进度权重 ─────────────────────────────────────────
PROGRESS_WEIGHTS = {
    "sampling": 20,
    "template": 30,
    "ocr": 35,
    "state_machine": 5,
    "refine": 10,
}
