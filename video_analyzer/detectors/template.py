from pathlib import Path

import cv2
import numpy as np

from video_analyzer.config import TEMPLATE_MATCH_THRESHOLD, TEMPLATE_SCORES, TEMPLATES_DIR
from video_analyzer.utils import empty_score

# 左侧列表在游戏内也可能部分匹配，提高阈值
TEMPLATE_THRESHOLDS = {
    "role_list_panel.png": 0.88,
}


class TemplateDetector:
    def __init__(self, templates_dir: Path | None = None):
        self.templates_dir = templates_dir or TEMPLATES_DIR
        self.templates: dict[str, np.ndarray] = {}
        self._load_templates()

    def _load_templates(self) -> None:
        for filename in TEMPLATE_SCORES:
            path = self.templates_dir / filename
            if not path.exists():
                continue
            img = cv2.imread(str(path))
            if img is not None:
                self.templates[filename] = img

    @property
    def loaded_count(self) -> int:
        return len(self.templates)

    @property
    def missing_templates(self) -> list[str]:
        return [name for name in TEMPLATE_SCORES if name not in self.templates]

    def match(self, frame: np.ndarray) -> tuple[list[str], dict[str, int]]:
        hits: list[str] = []
        score = empty_score()

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for name, template in self.templates.items():
            gray_tpl = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if template.ndim == 3 else template
            if gray_tpl.shape[0] > gray_frame.shape[0] or gray_tpl.shape[1] > gray_frame.shape[1]:
                continue

            result = cv2.matchTemplate(gray_frame, gray_tpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            threshold = TEMPLATE_THRESHOLDS.get(name, TEMPLATE_MATCH_THRESHOLD)
            if max_val >= threshold:
                hits.append(name.replace(".png", ""))
                for state, points in TEMPLATE_SCORES[name]:
                    score[state] += points

        return hits, score
