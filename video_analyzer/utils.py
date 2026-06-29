def format_time(seconds: float | int) -> str:
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_duration(seconds: int) -> str:
    m, s = divmod(max(0, seconds), 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h}小时{m}分{s}秒"
    return f"{m}分{s}秒"


def empty_score() -> dict[str, int]:
    return {"LOBBY": 0, "ROLE": 0, "PLAYING": 0, "ENDING": 0, "UNKNOWN": 0}


def round_iou(pred_start: int, pred_end: int, true_start: int, true_end: int) -> float:
    overlap = max(0, min(pred_end, true_end) - max(pred_start, true_start))
    union = max(pred_end, true_end) - min(pred_start, true_start)
    return overlap / union if union > 0 else 0.0
