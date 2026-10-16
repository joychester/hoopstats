"""第 2 步：多目标跟踪（ByteTrack 思路 + 镜头运动补偿 + 球衣颜色外观）。"""
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from frames import iter_frames


def to3(M):
    return np.vstack([np.array(M, np.float64), [0, 0, 1]])


def warp_box(b, A):
    pts = np.array([[b[0], b[1], 1], [b[2], b[3], 1]]).T
    q = (A @ pts)[:2].T
    return [q[0, 0], q[0, 1], q[1, 0], q[1, 1]]


def iou_matrix(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    a, b = np.array(a)[:, None], np.array(b)[None]
    ix = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    iy = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = ix * iy
    area = lambda x: (x[..., 2] - x[..., 0]) * (x[..., 3] - x[..., 1])
    return inter / (area(a) + area(b) - inter + 1e-6)


def torso_features(img, b):
    """取躯干区域（框高 18%~48%，宽中间 50%），返回 [深色占比, 白色占比, HS 直方图]。"""
    x1, y1, x2, y2 = b
    w, h = x2 - x1, y2 - y1
    crop = img[int(y1 + .18 * h):int(y1 + .48 * h), int(x1 + .25 * w):int(x2 - .25 * w)]
    if crop.size < 30:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    L, A, B = np.median(lab, 0)
    chroma = float(np.hypot(A - 128, B - 128))
    # 竖条纹（裁判服）：水平方向亮度梯度能量
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    stripe = float(np.abs(np.diff(g, axis=1)).mean()) if g.shape[1] > 2 else 0.0
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256]).flatten()
    hist /= hist.sum() + 1e-6
    head = img[int(y1):int(y1 + .15 * h), int(x1 + .3 * w):int(x2 - .3 * w)]
    hairL = float(np.median(cv2.cvtColor(head, cv2.COLOR_BGR2LAB)[..., 0])) if head.size >= 30 else -1.0
    return float(L), chroma, hist, stripe, hairL


class Track:
    _next = 1

    def __init__(self, box, score, fi, world, feat):
        self.id = Track._next
        Track._next += 1
        self.box, self.vel = list(box), np.zeros(4)
        self.hist = feat[2] if feat else None
        self.L = feat[0] if feat else None  # 躯干亮度滑动平均，用于防止串到对方球员身上
        self.obs = []  # (frame_idx, box, score, world_foot, (L, chroma, stripe, hairL))
        self.lost = 0
        self.end_reason = None  # 'crowd'（与同队球员重叠而结束）或 'lost'（拍不到/漏检）
        self.add(box, score, fi, world, feat)

    def add(self, box, score, fi, world, feat):
        if self.obs:
            self.vel = 0.6 * self.vel + 0.4 * (np.array(box) - np.array(self.box))
        self.box = list(box)
        self.lost = 0
        col = (feat[0], feat[1], feat[3], feat[4]) if feat else None  # (L, chroma, stripe, hairL)
        if feat:
            self.hist = feat[2] if self.hist is None else 0.85 * self.hist + 0.15 * feat[2]
            self.L = feat[0] if self.L is None else 0.85 * self.L + 0.15 * feat[0]
        self.obs.append((fi, [round(v, 1) for v in box], score, world, col))


def run(detjson, video, min_h=45):
    W, H = detjson["width"], detjson["height"]
    step = detjson["step"]
    reader = iter_frames(video, step)
    tracks, dead = [], []
    C = np.eye(3)  # 第 0 帧坐标 -> 当前帧坐标
    for k, fr in enumerate(detjson["frames"]):
        # next processed frame (frames are read in step with detection)
        for n, img in reader:
            if n >= fr["frame"]:
                break
        A = to3(fr["motion"])
        C = A @ C
        Cinv = np.linalg.inv(C)
        boxes, scores = [], []
        for c, s, *b in fr["dets"]:
            if c != 0:
                continue
            h = b[3] - b[1]
            # 前景观众：框贴着画面底边且很大
            if b[3] > H - 4 and h > 0.22 * H:
                continue
            if h < min_h:
                continue
            boxes.append(b)
            scores.append(s)
        feats = [torso_features(img, b) for b in boxes]
        worlds = []
        for b in boxes:
            p = Cinv @ np.array([(b[0] + b[2]) / 2, b[3], 1])
            worlds.append([round(p[0], 1), round(p[1], 1)])

        # 预测：镜头运动 + 匀速
        for t in tracks:
            t.box = warp_box(t.box, A)
            t.box = list(np.array(t.box) + t.vel)

        # 同队重叠：与另一个颜色相近的检测框重叠，或同时盖住两条颜色相近的轨迹。
        # 这时无法可靠区分是谁，宁可结束轨迹，也不冒跟错人的风险。
        same = lambda La, Lb: La is not None and Lb is not None and abs(La - Lb) <= 45
        crowded = [False] * len(boxes)
        dd = iou_matrix(boxes, boxes)
        tb = iou_matrix(boxes, [t.box for t in tracks])
        for i in range(len(boxes)):
            Li = feats[i][0] if feats[i] else None
            for j in range(len(boxes)):
                if i != j and dd[i, j] > 0.3 and same(Li, feats[j][0] if feats[j] else None):
                    crowded[i] = True
            if sum(1 for ti, t in enumerate(tracks) if tb[i, ti] > 0.3 and same(Li, t.L)) >= 2:
                crowded[i] = True

        hi = [i for i, s in enumerate(scores) if s >= 0.5]
        lo = [i for i, s in enumerate(scores) if s < 0.5]
        unmatched_t = list(range(len(tracks)))
        for group, thr in ((hi, 0.25), (lo, 0.4)):
            if not group or not unmatched_t:
                continue
            ious = iou_matrix([tracks[i].box for i in unmatched_t], [boxes[j] for j in group])
            cost = 1 - ious
            for a, ti in enumerate(unmatched_t):
                for bb, dj in enumerate(group):
                    if tracks[ti].hist is not None and feats[dj] is not None:
                        cost[a, bb] += 0.5 * cv2.compareHist(
                            tracks[ti].hist.astype(np.float32), feats[dj][2].astype(np.float32),
                            cv2.HISTCMP_BHATTACHARYYA)
                        # 颜色门槛：躯干亮度差太大（黑衣 vs 白衣）不允许匹配
                        if abs(tracks[ti].L - feats[dj][0]) > 45:
                            ious[a, bb] = 0
                            cost[a, bb] = 9
            r, c = linear_sum_assignment(cost)
            used_t, used_d = set(), set()
            for a, bb in zip(r, c):
                if ious[a, bb] >= thr:
                    ti, dj = unmatched_t[a], group[bb]
                    if crowded[dj]:
                        tracks[ti].end_reason = "crowd"
                    else:
                        tracks[ti].add(boxes[dj], scores[dj], k, worlds[dj], feats[dj])
                    used_t.add(ti)
                    used_d.add(dj)
            unmatched_t = [t for t in unmatched_t if t not in used_t]
            group[:] = [d for d in group if d not in used_d]
        for ti in unmatched_t:
            tracks[ti].lost += 1
        for dj in hi:  # 仅高置信度、未重叠的未匹配检测新建轨迹
            if not crowded[dj]:
                tracks.append(Track(boxes[dj], scores[dj], k, worlds[dj], feats[dj]))
        alive = []
        for t in tracks:
            if t.lost > 30 and t.end_reason is None:
                t.end_reason = "lost"
            (dead if t.end_reason else alive).append(t)
        tracks = alive
    for t in tracks:
        t.end_reason = t.end_reason or "lost"
    return dead + tracks
