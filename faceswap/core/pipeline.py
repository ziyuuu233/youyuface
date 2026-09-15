"""实时换脸流水线：摄像头采集 → 人脸检测 → 逐脸换脸 → 预览/虚拟摄像头输出。

设计：单一采集源。循环里读到一帧后，先换脸，再分别送 cv2 预览窗口
和（可选的）虚拟摄像头，避免重复打开摄像头导致设备占用。
"""

import threading
import time

import cv2
import numpy as np

from .face_analyser import get_many_faces


def swap_on_image(analyser, swapper, source_face, target_img):
    """静态图片换脸（用于无摄像头时验证模型链路）。"""
    out = target_img.copy()
    for face in get_many_faces(analyser, out):
        out = swapper.swap(out, source_face, face)
    return out


class LivePipeline:
    """摄像头实时预览。run() 在循环内逐帧处理，按 q 或 stop_event 退出。

    显示方式二选一：
    - show=True        : 用 cv2.imshow 弹独立窗口
    - on_frame 回调     : 把 (frame_bgr, fps) 交给调用方（通常是 GUI 内嵌显示）
    """

    def __init__(
        self,
        analyser,
        swapper,
        source_face,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        mirror: bool = True,
        virtual_cam: bool = False,
        virtual_cam_fps: float = 30.0,
        on_frame=None,
        match_skin: bool = True,
        skin_strength: float = 0.8,
        process_width: int = 0,
        skip_frames: bool = True,
    ):
        self.analyser = analyser
        self.swapper = swapper
        # 源脸用锁保护，支持运行中热切换
        self._source_lock = threading.Lock()
        self._source_face = source_face
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.mirror = mirror
        self.virtual_cam = virtual_cam
        self.virtual_cam_fps = virtual_cam_fps
        self.on_frame = on_frame
        self.match_skin = match_skin
        self.skin_strength = skin_strength
        # 处理分辨率：>0 时把帧缩放到该宽度再做人脸检测/换脸，输出再放大回去
        # 手机流建议设 640，可显著提升帧率
        self.process_width = process_width
        # 跳帧：处理不过来时丢弃旧帧，只处理最新帧，降低延迟
        self.skip_frames = skip_frames
        # 运行时状态，供 GUI 轮询显示
        self.status = {"fps": 0.0, "virtual_cam_active": False}
        # 内部标记是否为网络流（用于决定是否需要跳帧）
        self._is_url = isinstance(camera_index, str) and (
            camera_index.startswith(("http://", "https://", "rtsp://", "rtmp://"))
        )

    @property
    def source_face(self):
        with self._source_lock:
            return self._source_face

    def update_source_face(self, new_face):
        """运行中热切换源脸（线程安全）。"""
        with self._source_lock:
            self._source_face = new_face

    def _open_camera(self) -> cv2.VideoCapture:
        """打开摄像头。camera_index 支持：
        - 整数：本地摄像头编号（用 DSHOW，设置分辨率）
        - 字符串 URL：手机摄像头 App 输出的 HTTP MJPEG / RTSP 流
        """
        src = self.camera_index
        is_url = isinstance(src, str) and (
            src.startswith(("http://", "https://", "rtsp://", "rtmp://"))
        )

        if is_url:
            cap = cv2.VideoCapture(src)
            # 网络流关键优化：缓冲区只留 1 帧，避免旧帧堆积导致高延迟
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            # 本地摄像头：Windows 用 DSHOW 启动更快；MJPG 保证 720p 下高帧率
            idx = int(src)
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not cap.isOpened():
            raise RuntimeError(f"无法打开摄像头: {src}")
        return cap

    def _open_virtual_cam(self, frame_w: int, frame_h: int):
        """创建虚拟摄像头设备。返回 pyvirtualcam.Camera 或 None。"""
        import pyvirtualcam

        try:
            cam = pyvirtualcam.Camera(
                width=frame_w,
                height=frame_h,
                fps=self.virtual_cam_fps,
                fmt=pyvirtualcam.PixelFormat.BGR,
            )
            print(f"[vcam] 虚拟摄像头已启动: {cam.device} ({frame_w}x{frame_h})")
            return cam
        except Exception as e:  # noqa: BLE001 - 启动失败时仅降级为本地预览
            print(f"[vcam] 虚拟摄像头启动失败，将仅本地预览: {e}")
            return None

    def run(self, stop_event=None, show: bool = True):
        cap = self._open_camera()
        vcam = None
        fps = 0.0
        use_gui = self.on_frame is not None
        try:
            while stop_event is None or not stop_event.is_set():
                # 网络流跳帧：丢掉积压的旧帧，只取最新一帧，降低延迟
                if self._is_url and self.skip_frames:
                    # grab 多次丢弃缓冲区里的旧帧（grab 只解码不取出，速度快）
                    for _ in range(2):
                        if not cap.grab():
                            break
                    ok, frame = cap.retrieve()
                else:
                    ok, frame = cap.read()
                if not ok:
                    continue
                if self.mirror:
                    frame = cv2.flip(frame, 1)

                # 降分辨率处理：缩小后再换脸，输出放大回去（手机流显著提速）
                orig_h, orig_w = frame.shape[:2]
                if self.process_width > 0 and orig_w > self.process_width:
                    scale = self.process_width / orig_w
                    small = cv2.resize(
                        frame, (self.process_width, int(orig_h * scale)),
                        interpolation=cv2.INTER_AREA,
                    )
                else:
                    small = frame
                    scale = 1.0

                t0 = time.perf_counter()
                src = self.source_face  # 每帧只读一次（带锁），支持热切换
                original = small.copy()  # 肤色迁移需要原图作为参考
                for face in get_many_faces(self.analyser, small):
                    small = self.swapper.swap(small, src, face)
                    if self.match_skin:
                        try:
                            from .color_transfer import blend_skintone
                            small = blend_skintone(
                                original, small, face.bbox,
                                strength=self.skin_strength,
                            )
                        except Exception:  # noqa: BLE001 - 肤色迁移失败不影响换脸
                            pass
                elapsed = time.perf_counter() - t0
                fps = fps * 0.9 + 0.1 / elapsed if elapsed > 0 else fps

                # 放大回原始分辨率用于显示/输出
                if scale != 1.0:
                    frame = cv2.resize(small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                else:
                    frame = small

                # 首次拿到真实帧尺寸后再开虚拟摄像头（用实际分辨率而非请求值）
                if self.virtual_cam and vcam is None:
                    h, w = frame.shape[:2]
                    vcam = self._open_virtual_cam(w, h)
                    self.status["virtual_cam_active"] = vcam is not None

                # 推送到虚拟摄像头：BGR 帧，保证连续内存
                if vcam is not None:
                    vcam.send(np.ascontiguousarray(frame, dtype=np.uint8))

                # 显示：优先 GUI 内嵌回调，否则用 cv2 独立窗口
                if use_gui:
                    # 不在工作线程直接操作 Tk 控件，由回调内部用 root.after 切回主线程
                    self.on_frame(frame, fps, vcam is not None)
                elif show:
                    cv2.putText(
                        frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    )
                    if vcam is not None:
                        cv2.putText(
                            frame, "Vcam ON", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2,
                        )
                    cv2.imshow("faceswap live (q to quit)", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                self.status["fps"] = fps
        finally:
            cap.release()
            if vcam is not None:
                vcam.close()
            cv2.destroyAllWindows()
