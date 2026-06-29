"""角色揭示界面检测：识别固定「判定页面」，必须有「你已选择」锚点。"""

import cv2
import numpy as np

from video_analyzer.utils import empty_score

ROLE_TITLE_BAND = (0.28, 0.18, 0.72, 0.32)
ROLE_CENTER_ICON = (0.36, 0.28, 0.64, 0.52)


def _crop(frame: np.ndarray, region: tuple[float, float, float, float]) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = region
    return frame[int(h * y1) : int(h * y2), int(w * x1) : int(w * x2)]


def _bright_text_band_ratio(roi: np.ndarray) -> float:
    if roi.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray > 175))


def _center_icon_ratio(roi: np.ndarray) -> float:
    if roi.size == 0:
        return 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = (sat >= 100) & (val >= 90)
    return float(np.mean(mask))


def has_role_anchor(ocr_text: list[str], template_hits: list[str], visual_hits: list[str]) -> bool:
    """必须有「你已选择」类锚点，左侧列表单独出现不算。"""
    ocr = " ".join(ocr_text)
    if any(k in ocr for k in ("你已选择", "你的身份", "你是")):
        return True
    if "role_chosen_text" in template_hits:
        return True
    if "role_title_band" in visual_hits:
        return True
    return False


def detect_role_reveal(frame: np.ndarray) -> tuple[dict[str, int], list[str]]:
    """仅检测角色判定页专有特征，不单独用左侧列表计分。"""
    score = empty_score()
    hits: list[str] = []

    title_ratio = _bright_text_band_ratio(_crop(frame, ROLE_TITLE_BAND))
    icon_ratio = _center_icon_ratio(_crop(frame, ROLE_CENTER_ICON))

    if title_ratio >= 0.015:
        score["ROLE"] += 5
        hits.append("role_title_band")

    if title_ratio >= 0.015 and icon_ratio >= 0.08:
        score["ROLE"] += 3
        hits.append("center_role_icon")
        hits.append("role_reveal_layout")

    return score, hits


def apply_role_gate(entry: dict, min_state_score: int = 3) -> str:
    """ROLE 状态必须过锚点校验，否则降为 PLAYING/次高分。"""
    if entry.get("state") != "ROLE":
        return entry["state"]

    if has_role_anchor(
        entry.get("ocr_text", []),
        entry.get("template_hits", []),
        entry.get("visual_hits", []),
    ):
        return "ROLE"

    score = dict(entry.get("score", empty_score()))
    score["ROLE"] = 0
    alt = max(score, key=score.get)
    if score[alt] >= min_state_score:
        return alt
    return "PLAYING"


def is_strong_role_frame(frame: dict) -> bool:
    if frame.get("state") != "ROLE":
        return False
    return has_role_anchor(
        frame.get("ocr_text", []),
        frame.get("template_hits", []),
        frame.get("visual_hits", []),
    )
