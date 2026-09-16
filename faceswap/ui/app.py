"""Tkinter GUI：左侧源人脸缩略图与参数控件，右侧内嵌实时视频预览。

设计要点（线程安全 + 低卡顿）：
- 换脸流水线跑在后台线程，只把最新帧写入"帧槽"（带锁），不排队。
- 主线程用 after 定时轮询帧槽，有新帧才渲染，避免回调堆积。
- Canvas 图像 item 只创建一次，之后 itemconfig 复用，不每帧 delete/create。
"""

import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

PREVIEW_W = 640
PREVIEW_H = 360
UI_POLL_MS = 15  # 主线程轮询间隔（约 60fps 上限，实际跟随后台处理帧率）


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
        # 预览帧槽：后台线程只写最新帧，主线程定时取（只保留最新，不积压）
        self._frame_slot = None
        self._frame_slot_lock = threading.Lock()
        self._frame_seq = 0  # 帧序号，主线程用来判断有没有新帧
        self._last_rendered_seq = -1
        self._canvas_item = None  # 复用同一个 Canvas 图像 item
        self._last_fps_text = ""

        self._build_ui()
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        # 启动主线程 UI 轮询（一直存在，没新帧时不渲染）
        self.root.after(UI_POLL_MS, self._ui_poll)

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

        # 摄像头源：支持本地编号 0/1/2 或手机摄像头 URL
        ttk.Label(left, text="摄像头源（编号或手机流地址）").pack(anchor="w")
        self.camera_var = tk.StringVar(value="0")
        ttk.Entry(left, textvariable=self.camera_var, width=20).pack(fill="x", pady=(0, 2))
        ttk.Label(
            left,
            text="本地摄像头填 0/1/2；手机摄像头填 URL\n例如 http://192.168.1.5:8080/video",
            foreground="gray",
            font=("", 8),
            wraplength=260,
            justify="left",
        ).pack(anchor="w", pady=(0, 6))

        # 计算后端
        ttk.Label(left, text="计算后端").pack(anchor="w")
        self.provider_var = tk.StringVar(value=self.default_provider)
        ttk.Combobox(
            left, textvariable=self.provider_var,
            values=["auto", "cuda", "dml", "cpu"],
            state="readonly", width=8,
        ).pack(anchor="w", pady=(0, 6))

        # 处理分辨率（降分辨率可提升手机流帧率）
        ttk.Label(left, text="处理分辨率（手机流建议选 640）").pack(anchor="w")
        self.process_width_var = tk.StringVar(value="原始")
        ttk.Combobox(
            left, textvariable=self.process_width_var,
            values=["原始", "960", "640", "480"],
            state="readonly", width=8,
        ).pack(anchor="w", pady=(0, 6))

        # 跳帧开关
        self.skip_frames_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            left, text="网络流跳帧（降低延迟）",
            variable=self.skip_frames_var,
        ).pack(anchor="w", pady=(0, 6))

        # 虚拟摄像头开关
        self.virtual_cam_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            left, text="启用虚拟摄像头（推送到会议/直播软件）",
            variable=self.virtual_cam_var,
        ).pack(anchor="w", pady=4)

        # 肤色匹配开关 + 强度
        self.match_skin_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            left, text="肤色跟随目标（让换脸肤色贴合画面）",
            variable=self.match_skin_var,
        ).pack(anchor="w", pady=(8, 2))
        ttk.Label(left, text="肤色匹配强度").pack(anchor="w")
        self.skin_strength_var = tk.DoubleVar(value=0.8)
        ttk.Scale(
            left, from_=0.0, to=1.0, variable=self.skin_strength_var,
            orient="horizontal",
        ).pack(fill="x", pady=(0, 4))

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
    def _parse_process_width(self) -> int:
        v = self.process_width_var.get().strip()
        return 0 if v == "原始" else int(v)

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
                camera_index=self.camera_var.get().strip(),
                virtual_cam=self.virtual_cam_var.get(),
                on_frame=self._on_frame,
                match_skin=self.match_skin_var.get(),
                skin_strength=self.skin_strength_var.get(),
                process_width=self._parse_process_width(),
                skip_frames=self.skip_frames_var.get(),
            )
            self.pipeline = pipeline
            self.root.after(0, lambda: self.status.config(text="预览中"))
            pipeline.run(stop_event=self.stop_event, show=False)
            print("[GUI] 流水线已停止", flush=True)
            self.root.after(0, self._on_pipeline_done)
        except Exception as e:  # noqa: BLE001 - GUI 层捕获所有错误并提示
            import traceback
            traceback.print_exc()
            err_msg = str(e)  # 先把异常信息存成普通变量，避免 lambda 闭包访问不到 e
            self.root.after(0, lambda: self._on_pipeline_error(err_msg))

    def _on_frame(self, frame_bgr: np.ndarray, fps: float, vcam_on: bool):
        """后台线程回调：只覆盖帧槽（永远只保留最新一帧），不调度 UI。"""
        with self._frame_slot_lock:
            self._frame_slot = (frame_bgr, fps, vcam_on)
            self._frame_seq += 1

    def _ui_poll(self):
        """主线程定时执行：有新帧才渲染，避免回调堆积。"""
        try:
            with self._frame_slot_lock:
                seq = self._frame_seq
                slot = self._frame_slot
            if seq != self._last_rendered_seq and slot is not None:
                self._last_rendered_seq = seq
                self._render_preview(slot[0], slot[1], slot[2])
        finally:
            self.root.after(UI_POLL_MS, self._ui_poll)

    def _render_preview(self, frame_bgr: np.ndarray, fps: float, vcam_on: bool):
        # 运行中实时同步参数到 pipeline
        if self.pipeline is not None:
            self.pipeline.match_skin = self.match_skin_var.get()
            self.pipeline.skin_strength = self.skin_strength_var.get()
            self.pipeline.process_width = self._parse_process_width()
            self.pipeline.skip_frames = self.skip_frames_var.get()

        # 读取画布当前实际大小（窗口缩放后会变化），等比缩放并居中（contain）
        cw = self.preview_canvas.winfo_width()
        ch = self.preview_canvas.winfo_height()
        if cw < 10 or ch < 10:  # 尚未完成布局时的回退值
            cw, ch = PREVIEW_W, PREVIEW_H

        ih, iw = frame_bgr.shape[:2]
        scale = min(cw / iw, ch / ih)
        dw = max(1, int(iw * scale))
        dh = max(1, int(ih * scale))

        # BGR->RGB + 缩放一次完成（cv2 比 PIL 快），再包成 PhotoImage
        resized = cv2.resize(frame_bgr, (dw, dh), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self._preview_img = ImageTk.PhotoImage(Image.fromarray(rgb))  # 持有引用防 GC

        # 复用同一个 Canvas item，不每帧 delete/create（减少 GC 抖动）
        if self._canvas_item is None:
            self._canvas_item = self.preview_canvas.create_image(
                cw // 2, ch // 2, image=self._preview_img, anchor="center"
            )
        else:
            self.preview_canvas.coords(self._canvas_item, cw // 2, ch // 2)
            self.preview_canvas.itemconfig(self._canvas_item, image=self._preview_img)

        fps_text = f"FPS: {fps:.1f}" + ("  |  虚拟摄像头: 开" if vcam_on else "")
        if fps_text != self._last_fps_text:  # 文字没变就不刷新控件
            self._last_fps_text = fps_text
            self.fps_var.set(fps_text)

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
