"""下载运行所需模型。

下载内容：
1. inswapper_128_fp16.onnx —— 换脸模型（来自 Deep-Live-Cam 官方 HuggingFace）
2. buffalo_l.zip           —— InsightFace 人脸检测/识别模型包（GitHub 官方 Release）

每个文件提供多个镜像地址，逐个尝试。下载失败时可手动下载并按脚本提示放置。
"""

import io
import os
import sys
import urllib.request
import zipfile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
BUFFALO_DIR = os.path.join(MODELS_DIR, "models", "buffalo_l")

DOWNLOADS = {
    # 目标路径: [候选镜像地址]
    os.path.join(MODELS_DIR, "inswapper_128_fp16.onnx"): [
        "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx",
        "https://hf-mirror.com/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx",
    ],
    os.path.join(MODELS_DIR, "buffalo_l.zip"): [
        "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        "https://hf-mirror.com/public-data/insightface/resolve/main/models/buffalo_l.zip",
    ],
}


def download(urls, dst: str):
    for url in urls:
        try:
            print(f"[down] {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp, open(dst, "wb") as f:
                total = int(resp.headers.get("Content-Length", 0))
                done = 0
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r  {done / 1e6:.1f} / {total / 1e6:.1f} MB", end="")
            print(f"\n[ok] 已保存 {dst}")
            return True
        except Exception as e:  # noqa: BLE001 - 逐个镜像重试
            print(f"\n[fail] {e}")
    return False


def extract_buffalo(zip_path: str):
    os.makedirs(BUFFALO_DIR, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.endswith(".onnx"):
                # zip 内可能带或不带目录前缀，统一拍平放到 BUFFALO_DIR
                z.extract(name, BUFFALO_DIR)
    # 处理可能的嵌套目录
    for root, _dirs, files in os.walk(BUFFALO_DIR):
        for name in files:
            if name.endswith(".onnx") and root != BUFFALO_DIR:
                os.replace(
                    os.path.join(root, name), os.path.join(BUFFALO_DIR, name)
                )
    print(f"[ok] buffalo_l 模型已解压到 {BUFFALO_DIR}")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    failures = []

    for dst, urls in DOWNLOADS.items():
        if os.path.exists(dst):
            print(f"[skip] 已存在 {dst}")
            continue
        if not download(urls, dst):
            failures.append(dst)

    zip_path = os.path.join(MODELS_DIR, "buffalo_l.zip")
    if os.path.exists(zip_path):
        extract_buffalo(zip_path)
        os.remove(zip_path)

    if failures:
        print("\n以下模型下载失败，请手动下载后放到对应位置：")
        for p in failures:
            print(f"  {p}")
        sys.exit(1)
    print("\n全部模型就绪。")


if __name__ == "__main__":
    main()
