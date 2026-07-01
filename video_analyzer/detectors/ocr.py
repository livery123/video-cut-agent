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

logger = logging.getLogger(__name__)

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
    def __init__(self) -> None:
        self._reader: Callable[[np.ndarray], str] | None = None
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
            raise RuntimeError(
                "OCR 不可用：请安装 Tesseract 并配置 chi_sim 语言包"
            ) from exc

    @staticmethod
    def crop_roi(frame: np.ndarray, region: tuple[float, float, float, float]) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = region
        return frame[int(h * y1) : int(h * y2), int(w * x1) : int(w * x2)]

    def read_text(self, image: np.ndarray) -> str:
        if self._reader is None:
            return ""
        return self._reader(image)
