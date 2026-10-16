"""按外观（OSNet）把本队轨迹的小块自动归组：一组 ≈ 一名球员。

做法（已用 eval/ 下的标准答案评估，见 eval/chunk_eval.py）：
1. 每条轨迹切成约 0.5 秒的小块（混人的轨迹前后两半会落到不同组）
2. 每块取最大的 4 张截图，OSNet 特征取平均
3. 完全链接层次聚类；时间重叠、来自不同轨迹的小块禁止同组（同一时刻不可能是同一人）
"""
import cv2
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from frames import iter_frames
from reid import OSNet

CHUNK = 8          # 每块观测数（15fps 下约 0.53 秒）
THRESHOLD = 0.35   # 聚类距离阈值（短视频标准答案上 0.30~0.40 结果稳定）


def build_chunks(tracks, size=CHUNK):
    chunks = []
    for t in tracks:
        for s in range(0, len(t.obs), size):
            obs = t.obs[s:s + size]
            if len(obs) >= max(3, size // 2) or s == 0:
                chunks.append({"track": t.id, "obs": list(obs), "k0": obs[0][0], "k1": obs[-1][0]})
            elif chunks and chunks[-1]["track"] == t.id:   # 太短的尾巴并入上一块
                chunks[-1]["obs"] += obs
                chunks[-1]["k1"] = obs[-1][0]
    return chunks


def box_key(k, b):
    return (int(k), tuple(round(v, 1) for v in b))


def read_crops(video, detjson, obs_list, pad=0.1):
    """按顺序解码，只裁出需要的球员截图（不保存整帧，长视频也不会占满内存）。
    返回 {(处理帧序号, 框): 截图}"""
    want = {}
    for o in obs_list:
        want.setdefault(detjson["frames"][o[0]]["frame"], []).append(o)
    out = {}
    for i, img in iter_frames(video, want=list(want)):
        for o in want[i]:
            x1, y1, x2, y2 = o[1]
            w, h = x2 - x1, y2 - y1
            out[box_key(o[0], o[1])] = img[max(0, int(y1 - pad * h)):int(y2 + pad * h),
                                           max(0, int(x1 - pad * w)):int(x2 + pad * w)].copy()
    return out


def chunk_samples(c, per_chunk=4):
    return sorted(c["obs"], key=lambda o: -(o[1][3] - o[1][1]))[:per_chunk]


def embed_chunks(chunks, crops, osn=None):
    osn = osn or OSNet()
    F = []
    for c in chunks:
        e = osn([crops[box_key(o[0], o[1])] for o in chunk_samples(c)]).mean(0)
        F.append(e / (np.linalg.norm(e) + 1e-8))
    return np.array(F, np.float32)


def cluster(F, chunks, thr=THRESHOLD):
    if len(chunks) == 1:
        return np.array([1])
    D = 1 - F @ F.T
    np.fill_diagonal(D, 0)
    tid = np.array([c["track"] for c in chunks])
    k0 = np.array([c["k0"] for c in chunks])
    k1 = np.array([c["k1"] for c in chunks])
    ov = (k0[:, None] <= k1[None]) & (k0[None] <= k1[:, None]) & (tid[:, None] != tid[None])
    D = np.where(ov, 2.0, D)
    Z = linkage(squareform(np.clip(D, 0, None), checks=False), "complete")
    return fcluster(Z, thr, "distance")
