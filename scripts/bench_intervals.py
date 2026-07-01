"""抽帧间隔精度测试：对比基准 result.json，输出每局开始时间与误差。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from video_analyzer.config import COARSE_SAMPLE_INTERVAL, UPLOADS_DIR
from video_analyzer.pipeline import AnalysisPipeline

DEFAULT_VIDEO = UPLOADS_DIR / "task_1.mp4"
BASELINE_TASK = ROOT / "tasks" / "task_20260629_213440_438460" / "result.json"
DEFAULT_INTERVAL = COARSE_SAMPLE_INTERVAL
REPORT_PATH = ROOT / "bench_report.txt"


@dataclass
class RoundRef:
    round_id: int
    start_seconds: int
    start_time: str


@dataclass
class AlignRow:
    truth: RoundRef | None
    pred: RoundRef | None
    error_seconds: int | None
    status: str  # hit | missed | false_pos


def load_truth(path: Path) -> list[RoundRef]:
    if not path.exists():
        raise FileNotFoundError(f"未找到基准标注: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        RoundRef(
            round_id=r["round_id"],
            start_seconds=int(r["start_seconds"]),
            start_time=r["start_time"],
        )
        for r in data.get("rounds", [])
    ]


def resolve_video(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"找不到测试视频: {path}")
    return str(path.resolve())


def tolerance_for(interval: float) -> float:
    return max(interval * 1.5, 3.0)


def align_rounds(
    truth: list[RoundRef], predicted: list[RoundRef], tol: float
) -> list[AlignRow]:
    rows: list[AlignRow] = []
    used_pred: set[int] = set()

    for gt in truth:
        best_idx = -1
        best_err = tol + 1
        for i, pred in enumerate(predicted):
            if i in used_pred:
                continue
            err = abs(pred.start_seconds - gt.start_seconds)
            if err <= tol and err < best_err:
                best_err = err
                best_idx = i

        if best_idx >= 0:
            pred = predicted[best_idx]
            used_pred.add(best_idx)
            rows.append(
                AlignRow(
                    truth=gt,
                    pred=pred,
                    error_seconds=int(best_err),
                    status="hit",
                )
            )
        else:
            rows.append(
                AlignRow(truth=gt, pred=None, error_seconds=None, status="missed")
            )

    for i, pred in enumerate(predicted):
        if i in used_pred:
            continue
        rows.append(
            AlignRow(truth=None, pred=pred, error_seconds=None, status="false_pos")
        )

    return rows


def build_report(
    *,
    video: str,
    interval: float,
    elapsed: float,
    meta: dict,
    truth: list[RoundRef],
    predicted: list[RoundRef],
    align_rows: list[AlignRow],
) -> str:
    tol = tolerance_for(interval)
    hits = sum(1 for r in align_rows if r.status == "hit")
    missed = sum(1 for r in align_rows if r.status == "missed")
    false_pos = sum(1 for r in align_rows if r.status == "false_pos")
    errors = [r.error_seconds for r in align_rows if r.error_seconds is not None]
    avg_err = sum(errors) / len(errors) if errors else 0.0

    lines = [
        "抽帧间隔精度测试",
        "=" * 72,
        f"视频:     {video}",
        f"采样间隔: {interval}s  (容差 ±{tol:.0f}s)",
        f"耗时:     {elapsed:.1f}s",
        f"抽帧:     {meta['total_frames']}  命中帧: {meta['matched_frames']}",
        "",
        "── 汇总 ──",
        f"基准局数: {len(truth)}",
        f"预测局数: {len(predicted)}",
        f"命中: {hits}  漏检: {missed}  误报: {false_pos}  平均误差: {avg_err:.1f}s",
        "",
        "── 逐局对比（相对基准）──",
        "-" * 60,
    ]

    for i, row in enumerate(align_rows, start=1):
        if row.status == "false_pos":
            continue
        gt = row.truth
        assert gt is not None
        if row.pred:
            pred_part = f"{row.pred.start_time} ({row.pred.start_seconds}s)"
        else:
            pred_part = "-"
        if row.status == "hit":
            status = f"✓ 命中  误差 {row.error_seconds}s"
        else:
            status = "✗ 漏检"
        lines.append(
            f"  #{gt.round_id}  基准 {gt.start_time} ({gt.start_seconds}s)"
            f"  →  {pred_part}  {status}"
        )

    lines.extend(["", "── 预测明细（全部）──"])
    if predicted:
        for rnd in predicted:
            lines.append(
                f"  #{rnd.round_id}  {rnd.start_time}  ({rnd.start_seconds}s)"
            )
    else:
        lines.append("  (无)")

    false_rows = [r for r in align_rows if r.status == "false_pos"]
    if false_rows:
        lines.extend(["", "── 误报 ──"])
        for row in false_rows:
            assert row.pred is not None
            lines.append(
                f"  #{row.pred.round_id}  {row.pred.start_time}  "
                f"({row.pred.start_seconds}s)"
            )

    return "\n".join(lines)


def run_benchmark(
    interval: float,
    video_path: Path,
    baseline: Path,
    out_path: Path,
) -> str:
    video = resolve_video(video_path)
    truth = load_truth(baseline)

    print(f"开始测试: 间隔 {interval}s, 视频 {Path(video).name}", flush=True)
    pipeline = AnalysisPipeline()
    t0 = time.perf_counter()
    result = pipeline.run(video, sample_interval=interval)
    elapsed = time.perf_counter() - t0

    predicted = [
        RoundRef(
            round_id=r["round_id"],
            start_seconds=int(r["start_seconds"]),
            start_time=r["start_time"],
        )
        for r in result["rounds"]
    ]
    align_rows = align_rounds(truth, predicted, tolerance_for(interval))
    report = build_report(
        video=video,
        interval=interval,
        elapsed=elapsed,
        meta=result["meta"],
        truth=truth,
        predicted=predicted,
        align_rows=align_rows,
    )
    out_path.write_text(report, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="抽帧间隔精度测试")
    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_VIDEO,
        help=f"测试视频 (默认 {DEFAULT_VIDEO.name})",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL,
        help=f"采样间隔秒数 (默认 {DEFAULT_INTERVAL})",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=BASELINE_TASK,
        help="基准 result.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPORT_PATH,
        help="报告输出路径",
    )
    args = parser.parse_args()

    report = run_benchmark(
        args.interval, args.video, args.baseline, args.out
    )
    print(report, flush=True)
    print(f"\n已写入: {args.out}", flush=True)


if __name__ == "__main__":
    main()
