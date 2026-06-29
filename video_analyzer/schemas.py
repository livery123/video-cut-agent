from typing import Any

from pydantic import BaseModel, Field


from video_analyzer.config import COARSE_SAMPLE_INTERVAL


class AnalyzeRequestOptions(BaseModel):
    sample_interval: float = COARSE_SAMPLE_INTERVAL
    enable_ocr: bool = True
    enable_refine: bool = True
    game_type: str = "goose_goose_duck"


class TaskCreatedResponse(BaseModel):
    task_id: str
    status: str = "processing"


class TaskProgressResponse(BaseModel):
    task_id: str
    status: str
    progress: int = 0
    current_step: str = ""
    message: str = ""
    error_message: str | None = None


class RoundResult(BaseModel):
    round_id: int
    start_time: str
    end_time: str
    start_seconds: int
    end_seconds: int
    duration_seconds: int
    confidence: float
    start_reason: str
    end_reason: str
    warnings: list[str] = Field(default_factory=list)
    gap_to_next_round_seconds: int | None = None
    match_frames: int | None = None
    match_score_avg: float | None = None


class AnalysisResultResponse(BaseModel):
    task_id: str
    video_name: str
    duration: int
    sample_interval: float
    rounds: list[RoundResult]


class TimelineEntry(BaseModel):
    frame_id: int | None = None
    time: str
    seconds: int
    state: str
    confidence: float
    match_score: float | None = None
    ocr_text: list[str] = Field(default_factory=list)
    template_hits: list[str] = Field(default_factory=list)
    template_scores: dict[str, float] = Field(default_factory=dict)
    score: dict[str, int] = Field(default_factory=dict)
    skipped_ocr: bool = False


class TimelineResponse(BaseModel):
    total: int
    page: int
    page_size: int
    timeline: list[TimelineEntry]


class EvalMetrics(BaseModel):
    precision: float
    recall: float
    avg_boundary_error: float
    low_confidence_rounds: int
    details: list[dict[str, Any]] = Field(default_factory=list)
