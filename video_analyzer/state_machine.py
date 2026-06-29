from __future__ import annotations

from video_analyzer.config import (
    CONF_WEIGHTS,
    END_CONFIDENCE,
    END_OFFSET,
    ENDING_CONFIRM_FRAMES,
    MAX_ISOLATED_FRAMES,
    MAX_ROUND_DURATION,
    MIN_GAP_BETWEEN_ROUNDS,
    MIN_ROLE_SEGMENT_SECONDS,
    MIN_ROUND_DURATION,
    MIN_STATE_DURATION,
    MIN_STATE_SCORE,
    ROUND_COOLDOWN,
    START_CONFIDENCE,
    START_OFFSET,
)
from video_analyzer.detectors.role_reveal import is_strong_role_frame
from video_analyzer.utils import format_time


def _merge_role_segments(
    segments: list[tuple[int, int]], gap_seconds: int
) -> list[tuple[int, int]]:
    if not segments:
        return []
    merged = [segments[0]]
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= gap_seconds:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def _extract_role_segments(timeline: list[dict], sample_interval: float) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    i = 0
    while i < len(timeline):
        if timeline[i]["state"] == "ROLE" and is_strong_role_frame(timeline[i]):
            start = timeline[i]["seconds"]
            j = i + 1
            while j < len(timeline) and timeline[j]["state"] == "ROLE":
                j += 1
            end = timeline[j - 1]["seconds"]
            duration = end - start + int(sample_interval)
            if duration >= MIN_ROLE_SEGMENT_SECONDS:
                segments.append((start, end))
            i = j
        else:
            i += 1
    return _merge_role_segments(segments, MIN_GAP_BETWEEN_ROUNDS)


def pick_state(score: dict[str, int]) -> str:
    state = max(score, key=score.get)
    if score[state] < MIN_STATE_SCORE:
        return "UNKNOWN"
    return state


def smooth_timeline(timeline: list[dict], sample_interval: float) -> list[dict]:
    if len(timeline) < 3:
        return timeline

    states = [f["state"] for f in timeline]

    # 合并孤立状态
    for i in range(1, len(states) - 1):
        left, mid, right = states[i - 1], states[i], states[i + 1]
        if left == right and mid != left:
            run_len = 1
            j = i
            while j < len(states) and states[j] == mid:
                run_len += 0
                k = j
                while k < len(states) and states[k] == mid:
                    k += 1
                run_len = k - j
                break
            if run_len <= MAX_ISOLATED_FRAMES:
                for k in range(i, i + run_len):
                    states[k] = left

    # 合并过短片段
    idx = 0
    while idx < len(states):
        state = states[idx]
        j = idx
        while j < len(states) and states[j] == state:
            j += 1
        duration = (j - idx) * sample_interval
        min_dur = MIN_STATE_DURATION.get(state, 0)
        if duration < min_dur and j - idx < len(states):
            prev_state = states[idx - 1] if idx > 0 else None
            next_state = states[j] if j < len(states) else None
            replacement = prev_state or next_state or "UNKNOWN"
            for k in range(idx, j):
                states[k] = replacement
        idx = j

    # 可疑转移
    for i in range(1, len(states)):
        prev, curr = states[i - 1], states[i]
        if prev == "ENDING" and curr == "PLAYING":
            states[i] = "ENDING"
        elif prev == "ROLE" and curr == "LOBBY":
            states[i] = "UNKNOWN"

    for i, frame in enumerate(timeline):
        frame["state"] = states[i]
    return timeline


def _duration_score(duration: int) -> float:
    if duration < MIN_ROUND_DURATION:
        return 0.5
    if duration <= MAX_ROUND_DURATION:
        return 1.0
    return 0.6


def _continuity_score(timeline: list[dict], start: int, end: int) -> float:
    frames = [f for f in timeline if start <= f["seconds"] <= end]
    if not frames:
        return 0.5
    playing = sum(1 for f in frames if f["state"] == "PLAYING")
    ratio = playing / len(frames)
    return 0.5 if ratio < 0.3 else ratio


def calc_confidence(
    start_reason: str,
    end_reason: str,
    start_seconds: int,
    end_seconds: int,
    timeline: list[dict],
    start_transition: tuple[str, str] | None,
    end_type: str,
) -> float:
    start_conf = 0.65
    if start_transition:
        start_conf = START_CONFIDENCE.get(start_transition, 0.65)
        if start_transition[1] == "ROLE":
            start_conf = max(start_conf, 0.93)
    end_conf = END_CONFIDENCE.get(end_type, 0.70)
    duration = end_seconds - start_seconds
    d_score = _duration_score(duration)
    c_score = _continuity_score(timeline, start_seconds, end_seconds)
    return round(
        start_conf * CONF_WEIGHTS["start"]
        + end_conf * CONF_WEIGHTS["end"]
        + d_score * CONF_WEIGHTS["duration"]
        + c_score * CONF_WEIGHTS["continuity"],
        2,
    )


def detect_rounds(timeline: list[dict], video_duration: float, sample_interval: float) -> list[dict]:
    """以「角色揭示片段」起点作为每一局开始（用户主要关心 start）。"""
    if not timeline:
        return []

    segments = _extract_role_segments(timeline, sample_interval)
    rounds: list[dict] = []

    for idx, (seg_start, seg_end) in enumerate(segments):
        start_time = max(0, seg_start - START_OFFSET)
        end_time = min(video_duration, seg_end + END_OFFSET)
        duration = int(end_time - start_time)

        ocr_joined = ""
        for frame in timeline:
            if seg_start <= frame["seconds"] <= seg_start + int(sample_interval):
                ocr_joined = " ".join(frame.get("ocr_text", []))
                break

        start_reason = (
            "识别到角色揭示界面（你已选择）"
            if "你已选择" in ocr_joined
            else "识别到角色揭示界面"
        )
        conf = calc_confidence(
            start_reason,
            "角色揭示片段结束",
            int(start_time),
            int(end_time),
            timeline,
            ("UNKNOWN", "ROLE"),
            "ending_streak",
        )
        warnings: list[str] = []
        if duration < MIN_ROUND_DURATION:
            conf = round(conf * 0.5, 2)
            warnings.append("duration_too_short")
        if duration > MAX_ROUND_DURATION:
            conf = round(conf * 0.6, 2)
            warnings.append("duration_too_long")

        rounds.append(
            {
                "round_id": idx + 1,
                "start_time": format_time(start_time),
                "end_time": format_time(end_time),
                "start_seconds": int(start_time),
                "end_seconds": int(end_time),
                "duration_seconds": duration,
                "confidence": conf,
                "start_reason": start_reason,
                "end_reason": "角色揭示界面结束",
                "warnings": warnings,
            }
        )

    for i in range(len(rounds) - 1):
        gap = rounds[i + 1]["start_seconds"] - rounds[i]["end_seconds"]
        rounds[i]["gap_to_next_round_seconds"] = max(0, gap)

    return rounds
