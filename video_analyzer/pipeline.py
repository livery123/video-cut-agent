from __future__ import annotations

import logging
from typing import Callable

from video_analyzer.config import (
    COARSE_SAMPLE_INTERVAL,
    DEBUG_DIR,
    MATCH_FRAME_WIDTH,
    PROGRESS_WEIGHTS,
    ROUND_START_KEYWORD,
    ROUND_START_OCR_THRESHOLD,
    ROUND_START_ROI,
)
from video_analyzer.detectors.role_start import RoleStartDetector
from video_analyzer.sampler import sample_coarse
from video_analyzer.video_info import read_video_info

logger = logging.getLogger(__name__)


class AnalysisPipeline:
    """固定 ROI + OCR 识别「你已选择」，连续命中段首帧为局开始。"""

    def __init__(
        self,
        enable_ocr: bool = True,
        progress_callback: Callable[[int, str, str], None] | None = None,
        cancelled_check: Callable[[], bool] | None = None,
    ):
        self.progress_callback = progress_callback
        self.cancelled_check = cancelled_check or (lambda: False)
        self.detector = RoleStartDetector()

    def _report(self, progress: int, step: str, message: str = "") -> None:
        if self.progress_callback:
            self.progress_callback(progress, step, message)

    def _check_cancel(self) -> None:
        if self.cancelled_check():
            raise InterruptedError("任务已取消")

    def run(
        self,
        video_path: str,
        sample_interval: float = COARSE_SAMPLE_INTERVAL,
        enable_refine: bool = False,
        task_id: str | None = None,
    ) -> dict:
        info = read_video_info(video_path)
        debug_dir = DEBUG_DIR / task_id if task_id else None
        if debug_dir:
            debug_dir.mkdir(parents=True, exist_ok=True)

        self._report(PROGRESS_WEIGHTS["sampling"], "抽帧", "正在采样...")
        frames = sample_coarse(video_path, sample_interval, MATCH_FRAME_WIDTH)
        self._check_cancel()

        self._report(40, "OCR 识别", f"「{ROUND_START_KEYWORD}」，共 {len(frames)} 帧")
        hits = self.detector.scan_frames(frames)
        matched_count = sum(1 for h in hits if h.matched)
        self._check_cancel()

        self._report(70, "分析", f"命中 {matched_count} 帧，提取局开始...")
        rounds = self.detector.detect_round_starts(hits, sample_interval)

        timeline = self.detector.build_timeline(hits)

        if debug_dir:
            import cv2

            for item, hit in zip(frames, hits):
                if hit.matched:
                    out = debug_dir / f"{int(item.seconds):06d}_OCR_{hit.score:.2f}.jpg"
                    cv2.imwrite(str(out), item.image)

        self._report(100, "完成", f"识别到 {len(rounds)} 局")

        return {
            "video_name": info.name,
            "duration": int(info.duration),
            "sample_interval": sample_interval,
            "rounds": rounds,
            "timeline": timeline,
            "meta": {
                "mode": "ocr_role_chosen_text",
                "keyword": ROUND_START_KEYWORD,
                "roi": list(ROUND_START_ROI),
                "ocr_threshold": ROUND_START_OCR_THRESHOLD,
                "min_gap_seconds": 120,
                "matched_frames": matched_count,
                "total_frames": len(frames),
            },
        }
