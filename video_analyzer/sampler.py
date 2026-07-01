from dataclasses import dataclass

import cv2
import numpy as np

from video_analyzer.config import MATCH_FRAME_WIDTH


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


def sample_coarse(
    video_path: str,
    interval: float,
    target_width: int = MATCH_FRAME_WIDTH,
) -> list[SampledFrame]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_duration = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps

    frames: list[SampledFrame] = []
    frame_id = 0
    t = 0.0

    while t <= total_duration + 1e-6:
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
