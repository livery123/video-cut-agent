import logging
import os
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

try:
    from rapidfuzz import fuzz

    def _fuzzy_ratio(text: str, keyword: str) -> float:
        return float(fuzz.partial_ratio(text, keyword))
except ImportError:
    from difflib import SequenceMatcher

    def _fuzzy_ratio(text: str, keyword: str) -> float:
        if keyword in text:
            return 100.0
        kw_len = len(keyword)
        if kw_len == 0 or len(text) < kw_len:
            return SequenceMatcher(None, text, keyword).ratio() * 100
        best = 0.0
        for i in range(len(text) - kw_len + 1):
            chunk = text[i : i + kw_len]
            best = max(best, SequenceMatcher(None, chunk, keyword).ratio())
        return best * 100

from video_analyzer.config import (
    DARK_RATIO_THRESHOLD,
    FUZZY_MATCH_THRESHOLD,
    KEYWORDS,
    MIN_STATE_SCORE,
    OCR_SCORE_RULES,
    ROI_REGIONS,
)
from video_analyzer.utils import empty_score

logger = logging.getLogger(__name__)

# 常见安装路径（可被 .env 中的 TESSERACT_CMD / TESSERACT_SEARCH_PATHS 覆盖）
DEFAULT_TESSERACT_CANDIDATES = [
    Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
    Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    Path(r"E:\Tesseract-OCR\tesseract.exe"),
    Path(r"D:\Tesseract-OCR\tesseract.exe"),
]


def _normalize_exe(path: Path) -> Path:
    if path.is_dir():
        return path / "tesseract.exe"
    return path


def _tessdata_for_exe(exe: Path, project_tessdata: Path, project_chi: Path) -> str | None:
    if project_chi.exists():
        return str(project_tessdata)
    bundled = exe.parent / "tessdata"
    return str(bundled) if bundled.is_dir() else None


def _resolve_tesseract() -> tuple[str | None, str | None]:
    """返回 (tesseract_exe, tessdata_dir)。

    优先级：.env 显式配置 > 搜索路径列表 > PATH（exe 为 None）。
    """
    from video_analyzer.config import ROOT_DIR

    project_tessdata = ROOT_DIR / "tessdata"
    project_chi = project_tessdata / "chi_sim.traineddata"

    cmd = os.environ.get("TESSERACT_CMD", "").strip()
    tessdata_env = os.environ.get("TESSDATA_PREFIX", "").strip()

    if cmd:
        exe_path = _normalize_exe(Path(cmd))
        if exe_path.exists():
            tessdata = tessdata_env or _tessdata_for_exe(exe_path, project_tessdata, project_chi)
            return str(exe_path), tessdata

    candidates: list[Path] = list(DEFAULT_TESSERACT_CANDIDATES)
    extra = os.environ.get("TESSERACT_SEARCH_PATHS", "").strip()
    for item in extra.split(";"):
        item = item.strip()
        if not item:
            continue
        p = Path(item)
        candidates.append(_normalize_exe(p))
        if p.suffix.lower() != ".exe":
            candidates.append(p / "tesseract.exe")

    seen: set[str] = set()
    for raw in candidates:
        exe_path = _normalize_exe(raw)
        key = str(exe_path)
        if key in seen:
            continue
        seen.add(key)
        if exe_path.exists():
            return str(exe_path), _tessdata_for_exe(exe_path, project_tessdata, project_chi)

    if tessdata_env and Path(tessdata_env).is_dir():
        return None, tessdata_env

    return None, None


class OCRDetector:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._reader: Callable[[np.ndarray], str] | None = None
        if enabled:
            self._init_reader()

    def _init_reader(self) -> None:
        try:
            import pytesseract

            exe, tessdata = _resolve_tesseract()
            if exe:
                pytesseract.pytesseract.tesseract_cmd = exe
            if tessdata:
                os.environ["TESSDATA_PREFIX"] = tessdata
            elif not os.environ.get("TESSERACT_CMD") and not os.environ.get("TESSDATA_PREFIX"):
                if "TESSDATA_PREFIX" in os.environ:
                    del os.environ["TESSDATA_PREFIX"]

            version = pytesseract.get_tesseract_version()
            logger.info("OCR 引擎: pytesseract v%s @ %s", version, exe or "PATH")

            def read(img: np.ndarray) -> str:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return pytesseract.image_to_string(
                    rgb, lang="chi_sim", config="--psm 7"
                )

            self._reader = read
        except Exception as exc:
            logger.warning("OCR 不可用，将仅依赖模板匹配: %s", exc)
            self.enabled = False

    @staticmethod
    def crop_roi(frame: np.ndarray, region: tuple[float, float, float, float]) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = region
        return frame[int(h * y1) : int(h * y2), int(w * x1) : int(w * x2)]

    def extract_text(self, frame: np.ndarray, prioritize_role: bool = False) -> list[str]:
        if not self.enabled or self._reader is None:
            return []

        if prioritize_role:
            region_order = ["role_center", "role_left", "center", "bottom", "top"]
        else:
            region_order = ["role_center", "center", "bottom", "top", "role_left"]

        texts: list[str] = []
        for name in region_order:
            region = ROI_REGIONS.get(name)
            if not region:
                continue
            roi = self.crop_roi(frame, region)
            if roi.size == 0:
                continue
            try:
                raw = self._reader(roi).strip()
            except Exception as exc:
                logger.warning("OCR 单帧失败，跳过: %s", exc)
                continue
            if raw:
                texts.extend(line.strip() for line in raw.splitlines() if line.strip())
        return texts

    @staticmethod
    def fuzzy_hit(text: str, keyword: str) -> bool:
        return _fuzzy_ratio(text, keyword) >= FUZZY_MATCH_THRESHOLD

    def score_from_texts(self, texts: list[str]) -> dict[str, int]:
        score = empty_score()
        joined = " ".join(texts)

        for keywords, state, points in OCR_SCORE_RULES:
            if any(self.fuzzy_hit(joined, kw) or any(self.fuzzy_hit(t, kw) for t in texts) for kw in keywords):
                score[state] += points

        for state, words in KEYWORDS.items():
            for word in words:
                if self.fuzzy_hit(joined, word) or any(self.fuzzy_hit(t, word) for t in texts):
                    score[state] += 1

        return score

    @staticmethod
    def dark_ratio(frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray < 30) / 255.0)

    def should_run_ocr(
        self,
        template_hits: list[str],
        score: dict[str, int],
        dark_ratio: float,
        prev_state: str | None,
        curr_top_state: str | None,
        visual_hits: list[str] | None = None,
    ) -> bool:
        if not self.enabled:
            return False

        visual_hits = visual_hits or []
        role_templates = (
            "role_card",
            "role_chosen_text",
            "role_list_panel",
            "role_page_layout",
        )
        lobby_hits = any(h in template_hits for h in ("start_button", "ready_button"))
        role_hits = any(h in template_hits for h in role_templates)
        role_visual = any(
            h in visual_hits
            for h in ("role_reveal_layout", "role_title_band", "center_role_icon")
        )
        ending_hits = "victory_text" in template_hits

        if role_hits or role_visual:
            return True
        if lobby_hits or ending_hits:
            return True
        if dark_ratio >= DARK_RATIO_THRESHOLD:
            return True
        if max(score.values()) < MIN_STATE_SCORE:
            return True
        if prev_state and curr_top_state and prev_state != curr_top_state:
            return True
        return False
