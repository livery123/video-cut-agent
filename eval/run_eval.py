"""评估脚本 — 需要先在 eval/dataset/ 上传标注视频和 JSON"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from video_analyzer.config import EVAL_DATASET_DIR
from video_analyzer.pipeline import AnalysisPipeline
from video_analyzer.utils import round_iou


def load_ground_truth(json_path: Path) -> list[dict]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return data.get("rounds", [])


def match_rounds(predicted: list[dict], truth: list[dict], iou_threshold: float = 0.8) -> tuple[int, int, list[float]]:
    matched = 0
    errors: list[float] = []
    used = set()

    for pred in predicted:
        best_iou = 0.0
        best_idx = -1
        for i, gt in enumerate(truth):
            if i in used:
                continue
            iou = round_iou(pred["start_seconds"], pred["end_seconds"], gt["start"], gt["end"])
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_iou >= iou_threshold and best_idx >= 0:
            matched += 1
            used.add(best_idx)
            gt = truth[best_idx]
            errors.append(abs(pred["start_seconds"] - gt["start"]))
            errors.append(abs(pred["end_seconds"] - gt["end"]))

    return matched, len(truth), errors


def main() -> None:
    dataset_dir = EVAL_DATASET_DIR
    videos = sorted(dataset_dir.glob("*.mp4")) + sorted(dataset_dir.glob("*.mkv"))

    if not videos:
        print(f"未找到评估视频，请上传到: {dataset_dir}")
        print("并配套标注 JSON，例如 video_001.json:")
        print('  {"rounds": [{"start": 18, "end": 602}]}')
        return

    pipeline = AnalysisPipeline(enable_ocr=True)
    total_pred = total_truth = total_matched = 0
    all_errors: list[float] = []
    low_conf = 0

    for video in videos:
        label_path = video.with_suffix(".json")
        if not label_path.exists():
            print(f"跳过 {video.name}：缺少标注 {label_path.name}")
            continue

        truth = load_ground_truth(label_path)
        result = pipeline.run(str(video))
        predicted = result["rounds"]

        matched, truth_count, errors = match_rounds(predicted, truth)
        total_matched += matched
        total_pred += len(predicted)
        total_truth += truth_count
        all_errors.extend(errors)
        low_conf += sum(1 for r in predicted if r["confidence"] < 0.7)

        print(f"\n{video.name}: 预测 {len(predicted)} 局, 标注 {truth_count} 局, 匹配 {matched} 局")

    if total_pred == 0 and total_truth == 0:
        print("没有可评估的数据")
        return

    precision = total_matched / total_pred if total_pred else 0
    recall = total_matched / total_truth if total_truth else 0
    avg_err = sum(all_errors) / len(all_errors) if all_errors else 0

    print("\n========== 评估结果 ==========")
    print(f"局数精确率: {precision:.1%}")
    print(f"局数召回率: {recall:.1%}")
    print(f"平均边界误差: {avg_err:.1f} 秒")
    print(f"低置信度局数: {low_conf}")


if __name__ == "__main__":
    main()
