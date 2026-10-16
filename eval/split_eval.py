"""评估“轨迹内部换人检测”：对每条轨迹找外观突变点，突变足够大就切开。
标准答案：MIXED 的轨迹应该被切开，纯净轨迹不应该被切开。"""
import json
import os
import pickle
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reid import Embedder, OSNet  # noqa: E402


def track_embeddings(track, frames_cache, osn, every=2):
    obs = track.obs[::every]
    crops = [Embedder.crop(frames_cache[o[0]], o[1]) for o in obs]
    return obs, osn(crops)


def change_score(E, w=4):
    """每个候选切点：前 w 个和后 w 个特征均值的余弦距离；返回 (最大距离, 位置)。"""
    best = (0.0, None)
    for i in range(w, len(E) - w + 1):
        a, b = E[i - w:i].mean(0), E[i:i + w].mean(0)
        d = 1 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
        if d > best[0]:
            best = (float(d), i)
    return best


def main(evdir, video, detjson):
    gt = json.load(open(f"{evdir}/gt.json"))["tracks"]
    data = pickle.load(open(f"{evdir}/tracks.pkl", "rb"))
    dj = json.load(open(detjson))
    need = sorted({o[0] for t in data["ours"] for o in t.obs[::2]})
    cap = cv2.VideoCapture(video)
    fc = {}
    for k in need:
        cap.set(cv2.CAP_PROP_POS_FRAMES, dj["frames"][k]["frame"])
        fc[k] = cap.read()[1]
    osn = OSNet()
    rows = []
    for t in data["ours"]:
        obs, E = track_embeddings(t, fc, osn)
        d, i = change_score(E)
        cut_t = dj["frames"][obs[i][0]]["t"] if i is not None else None
        rows.append((t.id, gt.get(str(t.id)), round(d, 3), cut_t, len(obs)))
    rows.sort(key=lambda r: -r[2])
    for r in rows:
        print(f"#{r[0]:3d} {r[1]:6s} 突变={r[2]:.3f} 位置={r[3]}s 样本={r[4]}")
    mixed = [r[2] for r in rows if r[1] == "MIXED"]
    pure = [r[2] for r in rows if r[1] != "MIXED"]
    print(f"MIXED 最小突变 {min(mixed):.3f}；纯净最大突变 {max(pure):.3f}")
    json.dump(rows, open(f"{evdir}/split_report.json", "w"), ensure_ascii=False)


if __name__ == "__main__":
    main(*sys.argv[1:4])
