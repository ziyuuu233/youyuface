"""换脸推理：inswapper_128 模型封装。

原理：模型接收「目标帧上对齐后的目标人脸裁剪 + 源人脸嵌入向量」，
在目标位置生成换脸结果并按掩膜融合贴回原帧（paste_back 内置边缘羽化）。
"""

from insightface.model_zoo import model_zoo


class FaceSwapper:
    def __init__(self, model_path: str, providers: list):
        # 注意：model_zoo.get_model 只认 providers 参数（传 session 会被忽略，
        # 且默认 provider 是 CUDA，在 DirectML 环境下会错误回退）
        self.model = model_zoo.get_model(model_path, providers=providers)
        self.input_size = self.model.input_size[0] if hasattr(self.model, "input_size") else 128

    def swap(self, target_frame, source_face, target_face):
        """把源脸贴到 target_frame 的 target_face 位置，返回新帧。"""
        return self.model.get(
            img=target_frame,
            target_face=target_face,
            source_face=source_face,
            paste_back=True,
        )
