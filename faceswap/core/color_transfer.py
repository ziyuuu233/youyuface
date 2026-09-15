"""肤色迁移：让换脸后的肤色匹配目标原图的肤色分布。

原理：Reinhard 颜色迁移（LAB 空间均值+标准差匹配）。
在换脸后，对目标人脸 bbox 区域做颜色迁移，使换脸结果的肤色
与目标帧原图中该位置的肤色一致，减少"面具感"。

为了实时性能，采用：
- 仅在 bbox 内的椭圆人脸区域应用（用关键点估算中心）
- 羽化边缘避免硬边界
- 可配置强度（0=不迁移，1=完全匹配）
"""

import cv2
import numpy as np


def reinhard_transfer(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """把 src 的颜色分布迁移到 ref（LAB 空间均值+标准差匹配）。

    Args:
        src: 待迁移的图像（BGR）
        ref:  颜色参考图像（BGR）
    Returns:
        迁移后的 src（BGR, uint8）
    """
    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)

    src_mean, src_std = cv2.meanStdDev(src_lab)
    ref_mean, ref_std = cv2.meanStdDev(ref_lab)

    src_mean = src_mean.reshape(1, 1, 3)
    src_std = src_std.reshape(1, 1, 3)
    ref_mean = ref_mean.reshape(1, 1, 3)
    ref_std = ref_std.reshape(1, 1, 3)

    # 避免除零
    src_std = np.where(src_std < 1e-6, 1.0, src_std)
    out = (src_lab - src_mean) * (ref_std / src_std) + ref_mean
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)


def blend_skintone(frame: np.ndarray, swapped_frame: np.ndarray, bbox, strength: float = 0.8):
    """把 swapped_frame 中 bbox 区域的肤色迁移到 frame 原图对应区域，再羽化融合回去。

    Args:
        frame:          原始目标帧（未换脸），作为肤色参考
        swapped_frame:  换脸后的帧
        bbox:           目标人脸 bbox [x1, y1, x2, y2]
        strength:       迁移强度 0~1，1=完全匹配参考肤色
    Returns:
        肤色融合后的帧
    """
    if strength <= 0:
        return swapped_frame

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))

    roi_w = x2 - x1
    roi_h = y2 - y1
    if roi_w < 8 or roi_h < 8:
        return swapped_frame

    ref_roi = frame[y1:y2, x1:x2]
    swap_roi = swapped_frame[y1:y2, x1:x2].copy()

    # 颜色迁移
    transferred = reinhard_transfer(swap_roi, ref_roi)

    # 椭圆 mask（人脸大致呈椭圆），中心在 bbox 中心偏上（额头方向）
    mask = np.zeros((roi_h, roi_w), dtype=np.float32)
    cx = roi_w // 2
    cy = int(roi_h * 0.48)  # 椭圆中心略偏上，覆盖额头到下巴
    ax = int(roi_w * 0.46)
    ay = int(roi_h * 0.48)
    cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 1.0, -1)

    # 羽化：高斯模糊让边缘柔和
    blur_k = max(3, (min(roi_w, roi_h) // 6) | 1)  # 确保奇数
    mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)
    mask = np.clip(mask * strength, 0, 1)

    # 按 mask 融合迁移结果与换脸结果
    mask_3 = mask[:, :, np.newaxis]
    blended = (transferred.astype(np.float32) * mask_3
               + swap_roi.astype(np.float32) * (1 - mask_3))
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    out = swapped_frame.copy()
    out[y1:y2, x1:x2] = blended
    return out
