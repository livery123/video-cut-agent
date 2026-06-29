from dataclasses import dataclass

import cv2
import numpy as np

from video_analyzer.config import FRAME_WIDTH


@dataclass
class SampledFrame:
    frame_id: int
    seconds: float
    image: np.ndarray


def _resize_frame(frame: np.ndarray, target_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    scale = target_width / w
    return cv2.resize(frame, (target_width, int(h * scale)), interpolation=cv2.INTER_AREA)


def sample_at_intervals(
    video_path: str,
    interval: float,
    target_width: int = FRAME_WIDTH,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
) -> list[SampledFrame]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_duration = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
    end = end_seconds if end_seconds is not None else total_duration

    frames: list[SampledFrame] = []
    frame_id = 0
    t = start_seconds

    while t <= end + 1e-6:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, raw = cap.read()
        if not ok or raw is None:
            t += interval
            continue

        frame_id += 1
        frames.append(
            SampledFrame(
                frame_id=frame_id,
                seconds=round(t, 3),
                image=_resize_frame(raw, target_width),
            )
        )
        t += interval

    cap.release()
    return frames


def sample_coarse(video_path: str, interval: float, target_width: int = FRAME_WIDTH) -> list[SampledFrame]:
    return sample_at_intervals(video_path, interval, target_width)


def sample_refine_window(
    video_path: str,
    center_seconds: float,
    window: float,
    interval: float,
    target_width: int = FRAME_WIDTH,
) -> list[SampledFrame]:
    start = max(0.0, center_seconds - window)
    end = center_seconds + window
    return sample_at_intervals(video_path, interval, target_width, start, end)


def frame_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    hist_a = cv2.calcHist([a], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist_b = cv2.calcHist([b], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    return float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))
