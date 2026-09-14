"""Tkinter GUI：左侧源人脸缩略图与参数控件，右侧内嵌实时视频预览。

设计要点（线程安全）：
- 换脸流水线跑在后台线程，通过 on_frame 回调把 (frame_bgr, fps, vcam_on) 交给 GUI。
- 回调内不直接操作 Tk 控件，而是用 root.after(0, ...) 把渲染切回主线程。
- PhotoImage 必须持有引用，否则被 GC 回收导致画面空白。
"""

import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

PREVIEW_W = 640
PREVIEW_H = 360


class FaceSwapApp:
    def __init__(self, root: tk.Tk, default_provider: str = "auto"):
        self.root = root
        root.title("实时换脸")
        root.geometry("1000x720")

        self.source_path = None
        self.stop_event = None
        self.worker = None
        self.analyser = None
        self.swapper = None
        self.pipeline = None
        self.default_provider = default_provider
        # 必须持有 PhotoImage 引用，防止被 GC 回收
        self._preview_img = None
        self._source_thumb = None

        self._build_ui()
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------------- UI 布局 ----------------
    def _build_ui(self):
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill="both", expand=True)

        # ---- 左侧：源脸 + 控件 ----
        left = ttk.Frame(main, width=300)
        left.pack(side="left", fill="y", padx=(0, 8))
        left.pack_propagate(False)

        ttk.Label(left, text="源人脸图片", font=("", 10, "bold")).pack(anchor="w")
        self.source_canvas = tk.Canvas(left, width=240, height=240, bg="#2b2b2b", highlightthickness=1)
        self.source_canvas.pack(fill="x", pady=(2, 8))
        ttk.Button(left, text="选择源人脸图片", command=self._pick_source).pack(fill="x")

        ttk.Separator(left).pack(fill="x", pady=10)

        # 摄像头编号
        ttk.Label(left, text="摄像头编号").pack(anchor="w")
        self.camera_var = tk.StringVar(value="0")
        ttk.Spinbox(left, from_=0, to=9, textvariable=self.camera_var, width=5).pack(anchor="w", pady=(0, 6))

        # 计算后端
        ttk.Label(left, text="计算后端").pack(anchor="w")
        self.provider_var = tk.StringVar(value=self.default_provider)
        ttk.Combobox(
            left, textvariable=self.provider_var,
            values=["auto", "cuda", "dml", "cpu"],
            state="readonly", width=8,
        ).pack(anchor="w", pady=(0, 6))

        # 虚拟摄像头开关
        self.virtual_cam_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            left, text="启用虚拟摄像头（推送到会议/直播软件）",
            variable=self.virtual_cam_var,
        ).pack(anchor="w", pady=4)

        # 启动/停止
        btn_frame = ttk.Frame(left)
        btn_frame.pack(fill="x", pady=10)
        self.start_btn = ttk.Button(btn_frame, text="开始预览", command=self._start)
        self.start_btn.pack(side="left", expand=True, fill="x", padx=(0, 4))
        self.stop_btn = ttk.Button(btn_frame, text="停止", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", expand=True, fill="x", padx=(4, 0))

        # ---- 右侧：视频预览 ----
        right = ttk.Frame(main)
        right.pack(side="left", fill="both", expand=True)

        ttk.Label(right, text="实时预览", font=("", 10, "bold")).pack(anchor="w")
        self.preview_canvas = tk.Canvas(
            right, width=PREVIEW_W, height=PREVIEW_H, bg="#000", highlightthickness=1
        )
        self.preview_canvas.pack(fill="both", expand=True, pady=(2, 4))

        self.fps_var = tk.StringVar(value="FPS: --")
        ttk.Label(right, textvariable=self.fps_var).pack(anchor="w")

        # ---- 底部状态栏 ----
        self.status = ttk.Label(self.root, text="就绪", anchor="w", relief="sunken", padding=4)
        self.status.pack(side="bottom", fill="x")

    # ---------------- 源脸选择与缩略图 ----------------
    def _pick_source(self):
        path = filedialog.askopenfilename(
            title="选择源人脸图片",
            filetypes=[("图片", "*.jpg *.jpeg *.png *.webp"), ("所有文件", "*.*")],
        )
        if path:
            self.source_path = path
            self._draw_source_thumb(path)
            # 如果预览正在运行，热切换源脸
            if self.pipeline is not None and self.analyser is not None:
                try:
                    from main import make_source
                    new_face = make_source(self.analyser, path)
                    self.pipeline.update_source_face(new_face)
                    self.status.config(text="已切换源脸")
                except Exception as e:  # noqa: BLE001
                    messagebox.showerror("切换失败", str(e))

    def _draw_source_thumb(self, path: str):
        """把源脸图片等比缩放绘制到左侧 Canvas。"""
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
            # 等比缩放适配 240x240
            img.thumbnail((240, 240), Image.LANCZOS)
            self._source_thumb = ImageTk.PhotoImage(img)
            self.source_canvas.delete("all")
            cw, ch = 240, 240
            self.source_canvas.create_image(
                cw // 2, ch // 2, image=self._source_thumb, anchor="center"
            )
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("图片加载失败", str(e))

    # ---------------- 启动 / 停止 ----------------
    def _start(self):
        if not self.source_path:
            messagebox.showwarning("提示", "请先选择源人脸图片")
            return
        print("[GUI] 开始按钮被点击，启动流水线...", flush=True)
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status.config(text="正在加载模型，请稍候（约 10-20 秒）...")
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._run_pipeline, daemon=True)
        self.worker.start()

    def _run_pipeline(self):
        try:
            from main import load_models, make_source
            from faceswap.core.pipeline import LivePipeline

            print("[GUI] 加载模型中...", flush=True)
            analyser, swapper, providers = load_models(self.provider_var.get())
            self.analyser, self.swapper = analyser, swapper
            print("[GUI] 模型加载完成，提取源脸...", flush=True)
            source_face = make_source(analyser, self.source_path)
            print("[GUI] 源脸提取完成，启动摄像头...", flush=True)

            pipeline = LivePipeline(
                analyser, swapper, source_face,
                camera_index=int(self.camera_var.get()),
                virtual_cam=self.virtual_cam_var.get(),
                on_frame=self._on_frame,
            )
            self.pipeline = pipeline
            self.root.after(0, lambda: self.status.config(text="预览中"))
            pipeline.run(stop_event=self.stop_event, show=False)
            print("[GUI] 流水线已停止", flush=True)
            self.root.after(0, self._on_pipeline_done)
        except Exception as e:  # noqa: BLE001 - GUI 层捕获所有错误并提示
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self._on_pipeline_error(str(e)))

    def _on_frame(self, frame_bgr: np.ndarray, fps: float, vcam_on: bool):
        """流水线回调：把帧转成 PhotoImage 并通过 root.after 切回主线程渲染。"""
        # BGR -> RGB -> PIL -> 缩放到预览尺寸 -> PhotoImage
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img.thumbnail((PREVIEW_W, PREVIEW_H), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        # 必须在主线程更新控件
        self.root.after(0, lambda: self._render_preview(photo, fps, vcam_on))

    def _render_preview(self, photo, fps: float, vcam_on: bool):
        self._preview_img = photo  # 持有引用防 GC
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(
            PREVIEW_W // 2, PREVIEW_H // 2, image=photo, anchor="center"
        )
        self.fps_var.set(f"FPS: {fps:.1f}" + ("  |  虚拟摄像头: 开" if vcam_on else ""))

    def _stop(self):
        if self.stop_event:
            self.stop_event.set()

    def _on_pipeline_done(self):
        self.pipeline = None
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status.config(text="已停止")

    def _on_pipeline_error(self, msg: str):
        self._on_pipeline_done()
        self.status.config(text="出错")
        messagebox.showerror("错误", msg)

    def _on_close(self):
        self._stop()
        self.root.destroy()
