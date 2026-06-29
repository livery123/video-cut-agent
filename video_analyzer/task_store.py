import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from video_analyzer.config import TASKS_DIR


class TaskStore:
    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or TASKS_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cancelled: set[str] = set()

    def _task_dir(self, task_id: str) -> Path:
        path = self.base_dir / task_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def create_task(self, video_name: str, options: dict[str, Any]) -> str:
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        meta = {
            "task_id": task_id,
            "video_name": video_name,
            "status": "processing",
            "progress": 0,
            "current_step": "初始化",
            "message": "",
            "error_message": None,
            "options": options,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        self.save_meta(task_id, meta)
        return task_id

    def save_meta(self, task_id: str, meta: dict[str, Any]) -> None:
        with self._lock:
            meta["updated_at"] = datetime.now().isoformat()
            path = self._task_dir(task_id) / "meta.json"
            path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_meta(self, task_id: str) -> dict[str, Any] | None:
        path = self._task_dir(task_id) / "meta.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def update_progress(self, task_id: str, progress: int, step: str, message: str = "") -> None:
        meta = self.load_meta(task_id)
        if not meta:
            return
        meta["progress"] = progress
        meta["current_step"] = step
        meta["message"] = message
        self.save_meta(task_id, meta)

    def save_result(self, task_id: str, result: dict[str, Any]) -> None:
        path = self._task_dir(task_id) / "result.json"
        with self._lock:
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        meta = self.load_meta(task_id)
        if meta:
            meta["status"] = "completed"
            meta["progress"] = 100
            meta["current_step"] = "完成"
            self.save_meta(task_id, meta)

    def save_timeline(self, task_id: str, timeline: list[dict[str, Any]]) -> None:
        path = self._task_dir(task_id) / "timeline.json"
        with self._lock:
            path.write_text(json.dumps(timeline, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_result(self, task_id: str) -> dict[str, Any] | None:
        path = self._task_dir(task_id) / "result.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def load_timeline(self, task_id: str) -> list[dict[str, Any]]:
        path = self._task_dir(task_id) / "timeline.json"
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    def mark_failed(self, task_id: str, error: str) -> None:
        meta = self.load_meta(task_id)
        if meta:
            meta["status"] = "failed"
            meta["error_message"] = error
            self.save_meta(task_id, meta)

    def cancel(self, task_id: str) -> None:
        self._cancelled.add(task_id)
        meta = self.load_meta(task_id)
        if meta:
            meta["status"] = "cancelled"
            self.save_meta(task_id, meta)

    def is_cancelled(self, task_id: str) -> bool:
        return task_id in self._cancelled

    def delete_task(self, task_id: str) -> None:
        import shutil

        self._cancelled.add(task_id)
        task_dir = self.base_dir / task_id
        if task_dir.exists():
            shutil.rmtree(task_dir, ignore_errors=True)
