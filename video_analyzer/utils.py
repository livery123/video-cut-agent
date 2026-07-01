def format_time(seconds: float | int) -> str:
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def round_iou(pred_start: int, pred_end: int, true_start: int, true_end: int) -> float:
    overlap = max(0, min(pred_end, true_end) - max(pred_start, true_start))
    union = max(pred_end, true_end) - min(pred_start, true_start)
    return overlap / union if union > 0 else 0.0
