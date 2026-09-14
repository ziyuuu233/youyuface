"""人脸检测与识别：基于 InsightFace buffalo_l 模型包。

buffalo_l 包含：
- det_10g.onnx   SCRFD 人脸检测
- w600k_r50.onnx ArcFace 人脸识别（生成换脸所需的 normed_embedding）
"""

import numpy as np


def create_face_analyser(model_root: str, providers: list, det_size: int = 640):
    """创建检测器。model_root 下需存在 models/buffalo_l/*.onnx"""
    from insightface.app import FaceAnalysis

    analyser = FaceAnalysis(
        name="buffalo_l",
        root=model_root,
        allowed_modules=["detection", "recognition"],
        providers=providers,
    )
    # det_size 越小越快，精度略降；实时场景可改 320
    analyser.prepare(ctx_id=0, det_size=(det_size, det_size))
    return analyser


def get_many_faces(analyser, frame: np.ndarray) -> list:
    """检测画面中的所有人脸，按面积从大到小排序（主脸优先）。"""
    faces = analyser.get(frame)
    return sorted(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True,
    )


def get_one_face(analyser, frame: np.ndarray):
    """取画面中最大的一张脸，没有则返回 None。"""
    faces = get_many_faces(analyser, frame)
    return faces[0] if faces else None
