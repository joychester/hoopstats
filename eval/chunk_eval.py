"""评估“小块归组”：所有本队轨迹切成约 0.5 秒的小块，用 OSNet 特征统一层次聚类。
时间重叠、来自不同轨迹的小块不能归到同一组（同一时刻不可能是同一人）。
指标按时长加权：纯度、组数、教练操作次数（=组数 + 放错组的小块数）。"""
import json, os, pickle, sys
import cv2, numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reid import Embedder, OSNet  # noqa


def chunk_label(gt, tid, t0, t1):
    lab = gt["tracks"].get(str(tid))
    if lab != "MIXED":
        return lab
    for a, b, l in gt.get("mixed_segments", {}).get(str(tid), []):
        if t0 >= a and t1 <= b:
            return l
    return None  # 不确定，不计分


def build_chunks(tracks, fps, size):
    chunks = []
    for t in tracks:
        for s in range(0, len(t.obs), size):
            obs = t.obs[s:s + size]
            if len(obs) >= max(3, size // 2) or s == 0:
                chunks.append({"track": t.id, "obs": obs, "k0": obs[0][0], "k1": obs[-1][0]})
            elif chunks and chunks[-1]["track"] == t.id:   # 尾巴并入上一块
                chunks[-1]["obs"] += obs
                chunks[-1]["k1"] = obs[-1][0]
    return chunks


def cluster(F, chunks, thr, cont_bonus=0.0):
    n = len(chunks)
    D = 1 - F @ F.T
    np.fill_diagonal(D, 0)
    tid = np.array([c["track"] for c in chunks])
    k0 = np.array([c["k0"] for c in chunks]); k1 = np.array([c["k1"] for c in chunks])
    ov = (k0[:, None] <= k1[None]) & (k0[None] <= k1[:, None]) & (tid[:, None] != tid[None])
    D = np.where(ov, 2.0, D)
    if cont_bonus:  # 同一轨迹里相邻的小块，距离打折（时间连续性）
        for i in range(n - 1):
            if tid[i] == tid[i + 1]:
                D[i, i + 1] = D[i + 1, i] = max(0.0, D[i, i + 1] - cont_bonus)
    Z = linkage(squareform(np.clip(D, 0, None), checks=False), "complete")
    return fcluster(Z, thr, "distance")


def score(cl, labels, w):
    m = np.array([l is not None for l in labels])
    wrong = 0.0; wrong_n = 0
    for c in set(cl[m]):
        idx = np.where(m & (cl == c))[0]
        labs = [labels[i] for i in idx]
        tot = {}
        for i, l in zip(idx, labs):
            tot[l] = tot.get(l, 0) + w[i]
        best = max(tot, key=tot.get)
        wrong += sum(w[i] for i, l in zip(idx, labs) if l != best)
        wrong_n += sum(1 for l in labs if l != best)
    return 1 - wrong / w[m].sum(), wrong_n


def main(evdir, video, detjson, size=8):
    gt = json.load(open(f"{evdir}/gt.json"))
    data = pickle.load(open(f"{evdir}/tracks.pkl", "rb")); fps = data["fps"]
    dj = json.load(open(detjson))
    chunks = build_chunks(data["ours"], fps, size)
    need = sorted({o[0] for c in chunks for o in c["obs"]})
    cap = cv2.VideoCapture(video); fc = {}
    for k in need:
        cap.set(cv2.CAP_PROP_POS_FRAMES, dj["frames"][k]["frame"]); fc[k] = cap.read()[1]
    osn = OSNet(); F = []
    for c in chunks:
        obs = sorted(c["obs"], key=lambda o: -(o[1][3] - o[1][1]))[:4]
        e = osn([Embedder.crop(fc[o[0]], o[1]) for o in obs]).mean(0); F.append(e / np.linalg.norm(e))
    F = np.array(F)
    labels = [chunk_label(gt, c["track"], c["k0"] / fps, c["k1"] / fps) for c in chunks]
    w = np.array([len(c["obs"]) / fps for c in chunks])
    n_lab = sum(l is not None for l in labels)
    print(f"{len(chunks)} 个小块（{n_lab} 个有标准答案，共 {w[[l is not None for l in labels]].sum():.1f} 秒），"
          f"球员 {sorted(set(l for l in labels if l))}")
    res = []
    for bonus in (0.0, 0.1):
        for thr in np.arange(0.15, 0.61, 0.025):
            cl = cluster(F, chunks, thr, bonus)
            pur, wn = score(cl, labels, w)
            k = len(set(cl))
            res.append({"bonus": bonus, "thr": round(float(thr), 3), "groups": k, "purity": round(pur, 3),
                        "wrong_chunks": wn, "actions": k + wn})
    for r in res:
        if r["thr"] in (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5):
            print(r)
    best = min(res, key=lambda r: (r["actions"], -r["purity"]))
    print("最佳:", best)
    json.dump(res, open(f"{evdir}/chunk_report.json", "w"))


if __name__ == "__main__":
    main(*sys.argv[1:4])
