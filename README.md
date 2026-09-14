# 实时换脸 (Real-time Face Swap)

基于 InsightFace + inswapper 的本地实时换脸工具，支持摄像头实时预览、静态图片换脸，并可推流到虚拟摄像头（OBS / 会议软件可直接选用）。

![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## 功能特性

- **实时换脸**：打开摄像头即可把源人脸实时换到自己的脸上，延迟低
- **虚拟摄像头推流**：换脸画面可推送到系统虚拟摄像头，会议 / 直播软件可直接选用
- **静态图片换脸**：命令行一键验证模型链路
- **多计算后端**：自动选择 CUDA / DirectML GPU 加速，无 GPU 时退回 CPU
- **图形界面**：无参数启动即打开 Tkinter GUI，选图、开关摄像头、切换后端都在界面里完成

## 环境要求

- Windows 10/11
- Python 3.11（安装时勾选 *Add to PATH*）
- （可选）NVIDIA / AMD / Intel 显卡用于加速

> 默认安装 DirectML 版 onnxruntime，Windows 下免装 CUDA Toolkit 即可用 GPU 加速；如需 CUDA 后端，可自行替换安装 `onnxruntime-gpu`。

## 快速开始

1. 双击 `安装.bat` —— 自动创建虚拟环境、安装依赖、下载模型（首次约 1~2 GB 下载）
2. 双击 `启动.bat` —— 打开 GUI，选择一张源人脸照片即可开始

或使用命令行：

```bash
# 静态图片换脸（验证链路）
python main.py -s face.jpg -t photo.jpg -o out.jpg

# 摄像头实时预览（auto 自动选 GPU 后端）
python main.py -s face.jpg

# 指定 CUDA 后端 + 推流虚拟摄像头
python main.py -s face.jpg -p cuda --virtual-cam
```

## 模型说明

运行前需下载两个模型（`安装.bat` 会自动完成，也可手动运行 `python scripts/download_models.py`）：

| 模型 | 用途 | 来源 |
|---|---|---|
| `inswapper_128_fp16.onnx` | 人脸替换 | [Deep-Live-Cam (HuggingFace)](https://huggingface.co/hacksider/deep-live-cam) |
| `buffalo_l` | 人脸检测 / 识别 / 关键点 | [InsightFace 官方 Release](https://github.com/deepinsight/insightface) |

下载脚本内置国内镜像地址，下载失败时可按脚本提示手动放置。

## 项目结构

```
├── main.py                  # 程序入口（GUI / 命令行）
├── faceswap/
│   ├── core/
│   │   ├── face_analyser.py # 人脸检测与分析
│   │   ├── swapper.py       # inswapper 换脸封装
│   │   └── pipeline.py      # 实时流水线 / 静态换脸
│   └── ui/app.py            # Tkinter GUI
├── scripts/download_models.py
├── 安装.bat / 启动.bat       # Windows 一键脚本
└── requirements.txt
```

## 免责声明

本项目仅供 **学习研究与个人娱乐** 使用。请在合法合规的前提下使用，不得用于伪造他人身份、诽谤、诈骗、侵犯肖像权或任何违法行为。使用者需对生成内容承担全部责任；如视频中涉及他人面部，请事先征得对方同意。

## License

[MIT](LICENSE)
