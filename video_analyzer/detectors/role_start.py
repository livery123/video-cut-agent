"""在固定 ROI 内 OCR 识别「你已选择」，连续命中段首帧为局开始。"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np

from video_analyzer.config import (
    MIN_GAP_BETWEEN_ROUNDS,
    OCR_WORKERS,
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
    ocr_text: str = ""


@dataclass
class ScanStats:
    total_frames: int = 0
    ocr_frames: int = 0
    matched_frames: int = 0

    def as_dict(self) -> dict:
        return {
            "total_frames": self.total_frames,
            "ocr_frames": self.ocr_frames,
            "matched_frames": self.matched_frames,
        }


class RoleStartDetector:
    """固定区域 OCR，匹配「你已选择」。"""

    def __init__(
        self,
        roi: tuple[float, float, float, float] = ROUND_START_ROI,
        keyword: str = ROUND_START_KEYWORD,
        threshold: float = ROUND_START_OCR_THRESHOLD,
        workers: int = OCR_WORKERS,
    ):
        self.roi = roi
        self.keyword = keyword
        self.threshold = threshold
        self.workers = max(1, workers)
        self.ocr = OCRDetector()
        self.last_scan_stats = ScanStats()
        logger.info(
            "局开始检测: roi=%s keyword=%s threshold=%s workers=%s",
            roi,
            keyword,
            threshold,
            self.workers,
        )

    def read_roi_text(self, frame: np.ndarray) -> tuple[str, float]:
        roi = self.ocr.crop_roi(frame, self.roi)
        if roi.size == 0:
            return "", 0.0

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
        rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

        try:
            raw = self.ocr.read_text(rgb)
        except Exception as exc:
            logger.warning("OCR 单帧失败: %s", exc)
            return "", 0.0

        text = raw.strip().replace(" ", "").replace("\n", "")
        fuzzy = _fuzzy_ratio(text, self.keyword)
        return text, fuzzy

    def scan_frame(self, frame: np.ndarray, seconds: float) -> FrameHit:
        text, fuzzy = self.read_roi_text(frame)
        matched = fuzzy >= self.threshold
        return FrameHit(
            seconds=seconds,
            matched=matched,
            score=round(fuzzy / 100.0, 3),
            ocr_text=text,
        )

    def scan_frames(self, frames: list) -> list[FrameHit]:
        if not frames:
            self.last_scan_stats = ScanStats()
            return []

        if self.workers == 1:
            hits = [self.scan_frame(item.image, item.seconds) for item in frames]
        else:
            tasks = [(i, item.image, item.seconds) for i, item in enumerate(frames)]

            def _one(task: tuple[int, np.ndarray, float]) -> tuple[int, FrameHit]:
                idx, image, seconds = task
                return idx, self.scan_frame(image, seconds)

            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                result_map = dict(pool.map(_one, tasks))
            hits = [result_map[i] for i in range(len(frames))]

        matched = sum(1 for h in hits if h.matched)
        self.last_scan_stats = ScanStats(
            total_frames=len(frames),
            ocr_frames=len(frames),
            matched_frames=matched,
        )
        logger.info(
            "OCR 扫描: %d 帧, 命中 %d 帧, 并行×%d",
            len(frames),
            matched,
            self.workers,
        )
        return hits

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
                    "match_score": h.score,
                    "ocr_text": [h.ocr_text] if h.ocr_text else [],
                }
            )
        return timeline
