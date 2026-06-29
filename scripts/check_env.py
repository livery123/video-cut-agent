"""检查 Python 依赖与 Tesseract OCR 是否就绪。"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import video_analyzer.config  # noqa: F401 — 加载 .env

from video_analyzer.detectors.ocr import _resolve_tesseract


def main() -> int:
    ok = True
    print(f"项目目录: {ROOT}")
    print(f"Python:   {sys.executable} ({sys.version.split()[0]})")
    print()

    deps = [
        "fastapi",
        "uvicorn",
        "cv2",
        "numpy",
        "rapidfuzz",
        "pytesseract",
        "dotenv",
    ]
    print("Python 依赖:")
    for name in deps:
        try:
            __import__(name if name != "cv2" else "cv2")
            print(f"  [OK] {name}")
        except ImportError:
            print(f"  [FAIL] {name}")
            ok = False
    print()

    template = ROOT / "templates" / "role_chosen_text.png"
    print("模板:")
    if template.exists():
        print(f"  [OK] {template.name}")
    else:
        print(f"  [FAIL] 缺少 {template}")
        ok = False
    print()

    exe, tessdata = _resolve_tesseract()
    cmd_cfg = os.environ.get("TESSERACT_CMD", "").strip()
    tess_cfg = os.environ.get("TESSDATA_PREFIX", "").strip()
    print("Tesseract OCR:")
    if exe:
        print(f"  可执行文件: {exe}")
    elif cmd_cfg:
        print(f"  可执行文件: [未找到] {cmd_cfg}")
    else:
        print("  可执行文件: 未配置（请在 .env 设置 TESSERACT_CMD）")
    if tessdata:
        print(f"  语言包目录: {tessdata}")
        chi = Path(tessdata) / "chi_sim.traineddata"
        if chi.exists():
            print("  [OK] chi_sim.traineddata")
        else:
            print("  [FAIL] 缺少 chi_sim.traineddata")
            ok = False
    elif tess_cfg:
        print(f"  语言包目录: [未找到] {tess_cfg}")
        ok = False
    else:
        print("  语言包目录: 未配置（请在 .env 设置 TESSDATA_PREFIX）")
        ok = False

    try:
        import pytesseract

        if exe:
            pytesseract.pytesseract.tesseract_cmd = exe
        version = pytesseract.get_tesseract_version()
        print(f"  [OK] Tesseract v{version}")
    except Exception as exc:
        print(f"  [FAIL] 无法调用 Tesseract: {exc}")
        print("  提示: 安装 Tesseract 并勾选 chi_sim，然后编辑 .env")
        ok = False

    print()
    if ok:
        print("环境检查通过，可运行: python main.py")
        return 0
    print("环境检查未通过，请按 README 完成配置。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
