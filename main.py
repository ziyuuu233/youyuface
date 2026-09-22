"""实时换脸软件入口。

用法：
  python main.py                      # 打开 GUI
  python main.py -s face.jpg          # 直接用摄像头实时预览（auto 后端）
  python main.py -s face.jpg -p cuda  # 指定 CUDA 后端
  python main.py -s face.jpg -t photo.jpg -o out.jpg  # 静态图片换脸（验证模型链路）
"""

import argparse
import os
import sys

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
INSWAPPER_PATH = os.path.join(MODELS_DIR, "inswapper_128_fp16.onnx")

import platform

PROVIDER_PRIORITY = ["CUDAExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider"]

DISCLAIMER = """\
【免责声明 / 使用须知】

1. 本软件仅供学习、研究与技术交流使用。
2. 严禁将本软件用于任何违法违规活动，包括但不限于：
   - 诈骗、敲诈勒索
   - 诽谤、侮辱、侵犯他人名誉权
   - 未经同意换用他人肖像，侵犯肖像权与隐私权
   - 伪造身份、制作虚假证据
   - 制作、传播色情低俗内容
   - 其他违反法律法规或公序良俗的行为
3. 使用本软件换用真人肖像前，必须取得肖像权人的明确同意，
   并在发布、传播时标注"AI 换脸"等显著标识。
4. 因使用本软件产生的一切法律责任与后果，均由使用者自行承担，
   软件作者不承担任何连带责任。

使用本软件即表示您已阅读、理解并同意以上条款。
如不同意，请立即停止使用并关闭本软件。
"""


def show_disclaimer_gui(root) -> bool:
    """GUI 模式下弹出免责声明，用户必须同意才能继续。"""
    from tkinter import messagebox

    return messagebox.askyesno(
        "免责声明",
        DISCLAIMER,
        icon="warning",
    )


def show_disclaimer_cli() -> bool:
    """命令行模式下打印免责声明并要求确认。"""
    print(DISCLAIMER)
    try:
        ans = input("请输入 Y 表示同意以上条款后继续，其他任意键退出: ").strip().lower()
    except EOFError:
        return False
    return ans == "y"


def imread_unicode(path: str) -> np.ndarray:
    """支持中文/Unicode 路径的 cv2.imread 替代实现。"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except (OSError, ValueError):
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def resolve_providers(choice: str) -> list:
    """把 auto/cuda/dml/cpu 转成 onnxruntime 的 provider 列表。"""
    import onnxruntime

    available = onnxruntime.get_available_providers()
    if choice == "cpu":
        return ["CPUExecutionProvider"]
    if choice == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if choice == "dml":
        if platform.system() == "Windows":
            return ["DmlExecutionProvider"]
        # macOS/Linux 无 DirectML，退回 CPU
        return ["CPUExecutionProvider"]
    if choice == "coreml":
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    # auto：按优先级选第一个可用的 GPU 后端
    for p in PROVIDER_PRIORITY:
        if p in available:
            return [p, "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def load_models(provider_choice: str):
    """加载检测器与换脸模型。"""
    if not os.path.exists(INSWAPPER_PATH):
        raise RuntimeError(
            f"找不到换脸模型 {INSWAPPER_PATH}\n请先运行: python scripts/download_models.py"
        )
    from faceswap.core.face_analyser import create_face_analyser
    from faceswap.core.swapper import FaceSwapper

    providers = resolve_providers(provider_choice)
    print(f"[info] 计算后端: {providers[0]}")
    analyser = create_face_analyser(MODELS_DIR, providers)
    swapper = FaceSwapper(INSWAPPER_PATH, providers)
    return analyser, swapper, providers


def make_source(analyser, source_path: str):
    """读取源脸图片，提取主脸（含 normed_embedding，换脸外观来源）。"""
    img = imread_unicode(source_path)
    if img is None:
        raise RuntimeError(f"无法读取源图片: {source_path}")
    from faceswap.core.face_analyser import get_one_face

    face = get_one_face(analyser, img)
    if face is None:
        raise RuntimeError("源图片中未检测到人脸，请换一张更清晰的正脸照片")
    return face


def run_image_mode(args):
    import time

    from faceswap.core.pipeline import swap_on_image

    analyser, swapper, _ = load_models(args.provider)
    source_face = make_source(analyser, args.source)
    target = imread_unicode(args.target)
    if target is None:
        raise RuntimeError(f"无法读取目标图片: {args.target}")

    t0 = time.perf_counter()
    out = swap_on_image(analyser, swapper, source_face, target)
    elapsed = time.perf_counter() - t0

    output = args.output or "output.jpg"
    cv2.imwrite(output, out)
    print(f"[info] 完成，耗时 {elapsed:.2f}s，已保存到 {output}")


def run_live_mode(args):
    from faceswap.core.pipeline import LivePipeline

    analyser, swapper, _ = load_models(args.provider)
    source_face = make_source(analyser, args.source)
    pipeline = LivePipeline(
        analyser, swapper, source_face,
        camera_index=args.camera,
        virtual_cam=args.virtual_cam,
    )
    pipeline.run()


def main():
    parser = argparse.ArgumentParser(description="本地实时换脸")
    parser.add_argument("-s", "--source", help="源人脸图片路径")
    parser.add_argument("-t", "--target", help="目标图片路径（静态换脸测试）")
    parser.add_argument("-o", "--output", default="output.jpg", help="静态换脸输出路径")
    parser.add_argument("--camera", type=int, default=0, help="摄像头编号，默认 0")
    parser.add_argument(
        "--virtual-cam", action="store_true",
        help="同时把换脸画面推送到系统虚拟摄像头（会议/直播软件可选为摄像头输入）",
    )
    parser.add_argument(
        "-p", "--provider",
        choices=["auto", "cuda", "dml", "coreml", "cpu"],
        default="auto",
        help="计算后端：auto 自动选 GPU，否则退回 CPU",
    )
    args = parser.parse_args()

    # 无参数时打开 GUI
    if not args.source:
        import tkinter as tk

        from faceswap.ui.app import FaceSwapApp

        root = tk.Tk()
        root.withdraw()  # 先隐藏主窗口，等免责声明处理完再显示
        if not show_disclaimer_gui(root):
            print("您未同意免责条款，程序退出。")
            root.destroy()
            return
        root.deiconify()
        FaceSwapApp(root)
        root.mainloop()
        return

    # 命令行模式也需确认免责条款
    if not show_disclaimer_cli():
        print("您未同意免责条款，程序退出。")
        return

    if args.target:
        run_image_mode(args)
    else:
        run_live_mode(args)


if __name__ == "__main__":
    main()
