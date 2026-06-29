import logging
import shutil
import threading
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from video_analyzer.config import (
    COARSE_SAMPLE_INTERVAL,
    MAX_DURATION_SECONDS,
    MAX_FILE_SIZE_MB,
    ROOT_DIR,
    SUPPORTED_FORMATS,
    TASKS_DIR,
    UPLOADS_DIR,
)
from video_analyzer.pipeline import AnalysisPipeline
from video_analyzer.schemas import (
    AnalysisResultResponse,
    RoundResult,
    TaskCreatedResponse,
    TaskProgressResponse,
    TimelineEntry,
    TimelineResponse,
)
from video_analyzer.task_store import TaskStore

logger = logging.getLogger(__name__)

app = FastAPI(title="鹅鸭杀录屏分局检测", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

store = TaskStore()
_running: dict[str, threading.Thread] = {}


def _run_analysis(task_id: str, video_path: Path, options: dict) -> None:
    try:

        def on_progress(progress: int, step: str, message: str) -> None:
            store.update_progress(task_id, progress, step, message)

        pipeline = AnalysisPipeline(
            enable_ocr=options.get("enable_ocr", True),
            progress_callback=on_progress,
            cancelled_check=lambda: store.is_cancelled(task_id),
        )
        result = pipeline.run(
            str(video_path),
            sample_interval=options.get("sample_interval", COARSE_SAMPLE_INTERVAL),
            enable_refine=options.get("enable_refine", True),
            task_id=task_id,
        )
        store.save_timeline(task_id, result.pop("timeline"))
        result["task_id"] = task_id
        store.save_result(task_id, result)
    except InterruptedError:
        logger.info("任务 %s 已取消", task_id)
    except Exception as exc:
        logger.exception("任务 %s 失败", task_id)
        store.mark_failed(task_id, str(exc))


@app.get("/")
def index():
    static_index = ROOT_DIR / "static" / "index.html"
    if static_index.exists():
        return FileResponse(static_index)
    return {"message": "API 运行中，请访问 /docs"}


@app.post("/api/video/analyze-rounds", response_model=TaskCreatedResponse)
async def analyze_rounds(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    sample_interval: float = Form(COARSE_SAMPLE_INTERVAL),
    enable_ocr: bool = Form(True),
    enable_refine: bool = Form(True),
    game_type: str = Form("goose_goose_duck"),
):
    suffix = Path(file.filename or "").suffix.lower().lstrip(".")
    if suffix not in SUPPORTED_FORMATS:
        raise HTTPException(400, f"不支持的视频格式: {suffix}，支持: {SUPPORTED_FORMATS}")

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(400, f"文件过大: {size_mb:.1f}MB，上限 {MAX_FILE_SIZE_MB}MB")

    options = {
        "sample_interval": sample_interval,
        "enable_ocr": enable_ocr,
        "enable_refine": enable_refine,
        "game_type": game_type,
    }
    task_id = store.create_task(file.filename or "video.mp4", options)

    video_path = UPLOADS_DIR / f"{task_id}_{file.filename}"
    video_path.write_bytes(content)

    thread = threading.Thread(
        target=_run_analysis,
        args=(task_id, video_path, options),
        daemon=True,
    )
    _running[task_id] = thread
    thread.start()

    return TaskCreatedResponse(task_id=task_id, status="processing")


@app.get("/api/video/analyze-rounds/{task_id}", response_model=TaskProgressResponse)
def get_task_status(task_id: str):
    meta = store.load_meta(task_id)
    if not meta:
        raise HTTPException(404, "任务不存在")
    return TaskProgressResponse(
        task_id=meta["task_id"],
        status=meta["status"],
        progress=meta.get("progress", 0),
        current_step=meta.get("current_step", ""),
        message=meta.get("message", ""),
        error_message=meta.get("error_message"),
    )


@app.get("/api/video/analyze-rounds/{task_id}/result", response_model=AnalysisResultResponse)
def get_task_result(task_id: str):
    meta = store.load_meta(task_id)
    if not meta:
        raise HTTPException(404, "任务不存在")
    if meta["status"] not in ("completed",):
        raise HTTPException(409, f"任务未完成，当前状态: {meta['status']}")

    result = store.load_result(task_id)
    if not result:
        raise HTTPException(404, "结果不存在")

    return AnalysisResultResponse(
        task_id=task_id,
        video_name=result["video_name"],
        duration=result["duration"],
        sample_interval=result["sample_interval"],
        rounds=[RoundResult(**r) for r in result["rounds"]],
    )


@app.get("/api/video/analyze-rounds/{task_id}/timeline", response_model=TimelineResponse)
def get_task_timeline(
    task_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=10000),
    state: str | None = Query(None),
):
    meta = store.load_meta(task_id)
    if not meta:
        raise HTTPException(404, "任务不存在")

    timeline = store.load_timeline(task_id)
    if state:
        timeline = [t for t in timeline if t.get("state") == state.upper()]

    total = len(timeline)
    start = (page - 1) * page_size
    chunk = timeline[start : start + page_size]

    return TimelineResponse(
        total=total,
        page=page,
        page_size=page_size,
        timeline=[TimelineEntry(**t) for t in chunk],
    )


@app.get("/api/video/analyze-rounds/{task_id}/timeline/export")
def export_task_timeline(task_id: str):
    meta = store.load_meta(task_id)
    if not meta:
        raise HTTPException(404, "任务不存在")

    timeline = store.load_timeline(task_id)
    if not timeline:
        raise HTTPException(404, "时间线不存在")

    lines = [
        "frame_id\tseconds\ttime\tstate\tmatch_score\tmatched\ttemplate_hits\ttemplate_scores",
    ]
    for t in timeline:
        hits = ",".join(t.get("template_hits") or [])
        matched = "Y" if hits else "N"
        score = t.get("match_score", t.get("confidence", 0))
        tpl_scores = t.get("template_scores") or {}
        scores_str = ";".join(f"{k}:{v:.3f}" for k, v in tpl_scores.items())
        lines.append(
            f"{t.get('frame_id', '')}\t{t.get('seconds', '')}\t{t.get('time', '')}\t"
            f"{t.get('state', '')}\t{score:.3f}\t{matched}\t{hits}\t{scores_str}"
        )

    content = "\n".join(lines)
    export_path = TASKS_DIR / task_id / "frames_export.txt"
    export_path.parent.mkdir(parents=True, exist_ok=True)
    export_path.write_text(content, encoding="utf-8")
    return FileResponse(
        export_path,
        media_type="text/plain; charset=utf-8",
        filename=f"{task_id}_frames.txt",
    )


@app.delete("/api/video/analyze-rounds/{task_id}")
def cancel_task(task_id: str):
    meta = store.load_meta(task_id)
    if not meta:
        raise HTTPException(404, "任务不存在")
    store.cancel(task_id)
    store.delete_task(task_id)
    return {"task_id": task_id, "status": "cancelled"}


static_dir = ROOT_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
