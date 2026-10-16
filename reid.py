"""球员外观特征（用于把同一人的短轨迹归到一组）。

两类特征：
- cnn: MobileNetV2（ImageNet）最后一层特征图，整体 + 上/中/下三段分别池化
- parts: 手工部位颜色：头发、躯干、膝盖、鞋的 HSV 直方图
"""
import os

import cv2
import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def _l2(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-8)


class Embedder:
    def __init__(self, path=os.path.join(HERE, "models", "mnv2_feat.onnx")):
        avail = ort.get_available_providers()
        prov = [p for p in (("CoreMLExecutionProvider",) if os.environ.get("HOOPSTATS_COREML") == "1" else ()) + ("CPUExecutionProvider",) if p in avail]
        self.sess = ort.InferenceSession(path, providers=prov)
        self.out = "463"

    @staticmethod
    def crop(img, b, pad=0.05):
        x1, y1, x2, y2 = b
        w, h = x2 - x1, y2 - y1
        return img[max(0, int(y1 - pad * h)):int(y2 + pad * h), max(0, int(x1 - pad * w)):int(x2 + pad * w)]

    def cnn(self, crops):
        if not crops:
            return np.zeros((0, 4 * 1280), np.float32)
        x = np.stack([(cv2.resize(c, (128, 256))[..., ::-1].astype(np.float32) / 255 - MEAN) / STD
                      for c in crops]).transpose(0, 3, 1, 2)
        f = self.sess.run([self.out], {"input": x})[0]  # N,1280,8,4
        parts = [f.mean((2, 3)), f[:, :, 0:3].mean((2, 3)), f[:, :, 3:6].mean((2, 3)), f[:, :, 6:8].mean((2, 3))]
        return np.concatenate([_l2(p) for p in parts], 1) / 2.0


class OSNet:
    """专门的行人重识别模型（OSNet，高通 AI Hub 导出的 ONNX），输出 512 维特征。"""

    def __init__(self, path=os.path.join(HERE, "models", "osnet.onnx"), imagenet_norm=False):
        avail = ort.get_available_providers()
        prov = [p for p in (("CoreMLExecutionProvider",) if os.environ.get("HOOPSTATS_COREML") == "1" else ()) + ("CPUExecutionProvider",) if p in avail]
        self.sess = ort.InferenceSession(path, providers=prov)
        self.norm = imagenet_norm

    def __call__(self, crops):
        out = []
        for c in crops:
            x = cv2.resize(c, (128, 256))[..., ::-1].astype(np.float32) / 255
            if self.norm:
                x = (x - MEAN) / STD
            out.append(self.sess.run(None, {"image": x.transpose(2, 0, 1)[None]})[0][0])
        return _l2(np.array(out, np.float32)) if out else np.zeros((0, 512), np.float32)


def parts_color(crop):
    """头发/躯干/膝盖/鞋 四个部位的 HSV 直方图（各 8x3），拼接后归一化。"""
    h, w = crop.shape[:2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    regions = [(0.0, 0.14, .3, .7), (0.2, 0.45, .25, .75), (0.62, 0.78, .15, .85), (0.88, 1.0, .1, .9)]
    feats = []
    for ya, yb, xa, xb in regions:
        r = hsv[int(ya * h):max(int(ya * h) + 1, int(yb * h)), int(xa * w):max(int(xa * w) + 1, int(xb * w))]
        hist = cv2.calcHist([r], [0, 1, 2], None, [8, 2, 3], [0, 180, 0, 256, 0, 256]).flatten()
        feats.append(_l2(np.sqrt(hist)))
    return np.concatenate(feats) / 2.0
