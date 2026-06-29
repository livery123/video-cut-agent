from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass
class VideoInfo:
    path: str
    name: str
    duration: float
    fps: float
    width: int
    height: int
    frame_count: int


def read_video_info(video_path: str | Path) -> VideoInfo:
    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frame_count / fps if fps > 0 else 0.0
    cap.release()

    return VideoInfo(
        path=str(path),
        name=path.name,
        duration=duration,
        fps=fps,
        width=width,
        height=height,
        frame_count=frame_count,
    )
