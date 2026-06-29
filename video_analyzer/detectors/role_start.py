"""在固定 ROI 内 OCR 识别「你已选择」，连续命中段首帧为局开始。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from video_analyzer.config import (
    MIN_GAP_BETWEEN_ROUNDS,
    ROUND_START_KEYWORD,
    ROUND_START_OCR_THRESHOLD,
    ROUND_START_ROI,
)
from video_analyzer.detectors.ocr import OCRDetector, _fuzzy_ratio
from video_analyzer.utils import format_time

logger = logging.getLogger(__name__)


@dataclass
class FrameHit:
    seconds: float
    matched: bool
    score: float
    template_hits: list[str] = field(default_factory=list)
    template_scores: dict[str, float] = field(default_factory=dict)
    ocr_text: str = ""


class RoleStartDetector:
    """固定区域 OCR，匹配「你已选择」。"""

    HIT_NAME = "role_chosen_text"

    def __init__(
        self,
        roi: tuple[float, float, float, float] = ROUND_START_ROI,
        keyword: str = ROUND_START_KEYWORD,
        threshold: float = ROUND_START_OCR_THRESHOLD,
    ):
        self.roi = roi
        self.keyword = keyword
        self.threshold = threshold
        self.ocr = OCRDetector(enabled=True)
        if not self.ocr.enabled:
            raise RuntimeError(
                "OCR 不可用：请安装 Tesseract（C:\\Program Files\\Tesseract-OCR），"
                "并确保 tessdata/chi_sim.traineddata 存在"
            )
        logger.info("局开始检测: OCR roi=%s keyword=%s threshold=%s", roi, keyword, threshold)

    @property
    def template_names(self) -> list[str]:
        return [self.HIT_NAME]

    def read_roi_text(self, frame: np.ndarray) -> tuple[str, float]:
        roi = self.ocr.crop_roi(frame, self.roi)
        if roi.size == 0 or self.ocr._reader is None:
            return "", 0.0

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
        rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

        try:
            raw = self.ocr._reader(rgb)
        except Exception as exc:
            logger.warning("OCR 单帧失败: %s", exc)
            return "", 0.0

        text = raw.strip().replace(" ", "").replace("\n", "")
        fuzzy = _fuzzy_ratio(text, self.keyword)
        return text, fuzzy

    def scan_frame(self, frame: np.ndarray, seconds: float) -> FrameHit:
        text, fuzzy = self.read_roi_text(frame)
        matched = fuzzy >= self.threshold
        score = round(fuzzy / 100.0, 3)
        return FrameHit(
            seconds=seconds,
            matched=matched,
            score=score,
            template_hits=[self.HIT_NAME] if matched else [],
            template_scores={self.HIT_NAME: score},
            ocr_text=text,
        )

    def scan_frames(self, frames: list) -> list[FrameHit]:
        return [self.scan_frame(item.image, item.seconds) for item in frames]

    def detect_round_starts(
        self,
        hits: list[FrameHit],
        sample_interval: float,
    ) -> list[dict]:
        if not hits:
            return []

        groups: list[list[FrameHit]] = []
        group: list[FrameHit] = []
        max_gap = sample_interval * 2.5

        for hit in hits:
            if not hit.matched:
                if group:
                    groups.append(group)
                    group = []
                continue
            if group and hit.seconds - group[-1].seconds > max_gap:
                groups.append(group)
                group = [hit]
            else:
                group.append(hit)
        if group:
            groups.append(group)

        candidates: list[dict] = []
        for group in groups:
            first = group[0]
            start_sec = int(first.seconds)
            score_avg = float(np.mean([h.score for h in group]))
            candidates.append(
                {
                    "round_id": 0,
                    "start_time": format_time(start_sec),
                    "start_seconds": start_sec,
                    "end_time": format_time(group[-1].seconds),
                    "end_seconds": int(group[-1].seconds),
                    "duration_seconds": int(group[-1].seconds - start_sec),
                    "confidence": round(min(0.99, score_avg), 2),
                    "start_reason": "OCR 识别到「你已选择」",
                    "end_reason": "",
                    "warnings": [],
                    "match_frames": len(group),
                    "match_score_avg": round(score_avg, 3),
                    "matched_templates": [self.HIT_NAME],
                }
            )

        rounds: list[dict] = []
        for rnd in candidates:
            if rounds and rnd["start_seconds"] - rounds[-1]["start_seconds"] < MIN_GAP_BETWEEN_ROUNDS:
                continue
            rnd["round_id"] = len(rounds) + 1
            rounds.append(rnd)

        return rounds

    def build_timeline(self, hits: list[FrameHit]) -> list[dict]:
        timeline = []
        for i, h in enumerate(hits, start=1):
            timeline.append(
                {
                    "frame_id": i,
                    "seconds": int(h.seconds),
                    "time": format_time(h.seconds),
                    "state": "ROLE_START" if h.matched else "OTHER",
                    "confidence": h.score,
                    "template_hits": h.template_hits,
                    "match_score": h.score,
                    "template_scores": h.template_scores,
                    "ocr_text": [h.ocr_text] if h.ocr_text else [],
                    "score": {"ROLE": int(h.matched), "OTHER": int(not h.matched)},
                }
            )
        return timeline
