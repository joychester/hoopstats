"""YOLOX ONNX 检测器：检测人（COCO 0）和球（COCO 32 sports ball）。"""
import os

import cv2
import numpy as np
import onnxruntime as ort

PERSON, BALL = 0, 32


class Detector:
    def __init__(self, path, size=640, threads=0):
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        # 默认用 CPU；设置环境变量 HOOPSTATS_COREML=1 可尝试 Core ML 加速（Apple 芯片的 Mac 上更有效）
        avail = ort.get_available_providers()
        providers = [p for p in (("CoreMLExecutionProvider",) if os.environ.get("HOOPSTATS_COREML") == "1" else ()) + ("CPUExecutionProvider",) if p in avail]
        self.sess = ort.InferenceSession(path, so, providers=providers)
        self.inp = self.sess.get_inputs()[0].name
        self.size = size
        grids, strides = [], []
        for s in (8, 16, 32):
            n = size // s
            xv, yv = np.meshgrid(np.arange(n), np.arange(n))
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((n * n, 1), s))
        self.grids = np.concatenate(grids).astype(np.float32)
        self.strides = np.concatenate(strides).astype(np.float32)

    def _run(self, img):
        h, w = img.shape[:2]
        r = min(self.size / h, self.size / w)
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        canvas[: int(h * r), : int(w * r)] = cv2.resize(img, (int(w * r), int(h * r)))
        x = canvas.transpose(2, 0, 1)[None].astype(np.float32)
        out = self.sess.run(None, {self.inp: x})[0][0]
        out[:, :2] = (out[:, :2] + self.grids) * self.strides
        out[:, 2:4] = np.exp(out[:, 2:4]) * self.strides
        return out, r

    def detect(self, img, conf_person=0.35, conf_ball=0.15):
        out, r = self._run(img)
        cls_scores = out[:, 5:] * out[:, 4:5]
        res = []
        for cls, thr in ((PERSON, conf_person), (BALL, conf_ball)):
            sc = cls_scores[:, cls]
            keep = sc > thr
            if not keep.any():
                continue
            b = out[keep, :4] / r
            s = sc[keep]
            xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                             b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
            idx = cv2.dnn.NMSBoxes(
                [[float(a), float(c), float(e - a), float(f - c)] for a, c, e, f in xyxy],
                s.tolist(), thr, 0.45)
            for i in np.array(idx).flatten():
                res.append((cls, float(s[i]), *xyxy[i].tolist()))
        return res  # list of (cls, score, x1, y1, x2, y2)
