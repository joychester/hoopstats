"""用标准答案评估“自动归组”效果。

评估单位：把每条纯净轨迹切成约 1 秒的小段（模拟真实场景里的短轨迹）。
指标：
- AUC：任取两段（来自不同原始轨迹），特征距离区分“同一人 / 不同人”的能力
- 最近邻准确率：每段最像的另一段（不同原始轨迹）是否同一人
- 聚类：阈值扫描，报告组数、纯度、教练操作次数（=给每组命名 + 把放错组的段移走）
"""
import json
import os
import pickle
import sys

import cv2
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reid import Embedder, OSNet, parts_color  # noqa: E402


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    # 距离越小越可能是同一人：AUC = P(d_pos < d_neg)
    return float((pos[:, None] < neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean())


def main(evdir, video, detjson, chunk=15, per_chunk=5):
    gt = json.load(open(f"{evdir}/gt.json"))["tracks"]
    data = pickle.load(open(f"{evdir}/tracks.pkl", "rb"))
    dj = json.load(open(detjson))
    tracks = [t for t in data["ours"] if gt.get(str(t.id), "MIXED") != "MIXED"]
    # 切块：每块 chunk 个观测，取其中最大的 per_chunk 个框
    items = []
    for t in tracks:
        for s in range(0, len(t.obs), chunk):
            obs = t.obs[s:s + chunk]
            if len(obs) < 5:
                continue
            obs = sorted(obs, key=lambda o: -(o[1][3] - o[1][1]))[:per_chunk]
            items.append({"track": t.id, "label": gt[str(t.id)], "obs": obs})
    # 读帧取截图
    need = sorted({o[0] for it in items for o in it["obs"]})
    cap = cv2.VideoCapture(video)
    frames = {}
    for k in need:
        cap.set(cv2.CAP_PROP_POS_FRAMES, dj["frames"][k]["frame"])
        frames[k] = cap.read()[1]
    emb = Embedder()
    osn, osn_n = OSNet(), OSNet(imagenet_norm=True)
    feats = {"cnn": [], "parts": [], "cnn+parts": [], "osnet": [], "osnet(归一化)": [], "osnet+parts": []}
    for it in items:
        crops = [Embedder.crop(frames[o[0]], o[1]) for o in it["obs"]]
        c = emb.cnn(crops).mean(0)
        p = np.mean([parts_color(x) for x in crops], 0)
        c, p = c / np.linalg.norm(c), p / np.linalg.norm(p)
        feats["cnn"].append(c)
        feats["parts"].append(p)
        feats["cnn+parts"].append(np.concatenate([c, p]) / np.sqrt(2))
        o = osn(crops).mean(0); o /= np.linalg.norm(o)
        on = osn_n(crops).mean(0); on /= np.linalg.norm(on)
        feats["osnet"].append(o)
        feats["osnet(归一化)"].append(on)
        feats["osnet+parts"].append(np.concatenate([o, p]) / np.sqrt(2))
    span = np.array([[it["obs"][0][0], it["obs"][-1][0]] for it in items])
    for it in items:
        ks = [o[0] for o in it["obs"]]
        it["span"] = (min(ks), max(ks))
    labels = np.array([it["label"] for it in items])
    tids = np.array([it["track"] for it in items])
    print(f"评估集：{len(tracks)} 条纯净轨迹 → {len(items)} 个约 1 秒的小段，{len(set(labels))} 名球员")
    report = {}
    sp = np.array([it["span"] for it in items])
    overlap = (sp[:, None, 0] <= sp[None, :, 1]) & (sp[None, :, 0] <= sp[:, None, 1])
    for name in list(feats):
        feats[name + "+约束"] = feats[name]
    for name, F in feats.items():
        F = np.array(F)
        D = squareform(pdist(F, "cosine"))
        if name.endswith("约束"):
            D = np.where(overlap & (tids[:, None] != tids[None]), 2.0, D)
        cross = tids[:, None] != tids[None]
        same = labels[:, None] == labels[None]
        iu = np.triu_indices(len(items), 1)
        m = cross[iu]
        a = auc(D[iu][m & same[iu]], D[iu][m & ~same[iu]])
        # 最近邻（排除同一原始轨迹）
        Dn = D.copy()
        Dn[~cross] = np.inf
        nn = Dn.argmin(1)
        valid = np.isfinite(Dn.min(1))
        nn_acc = float((labels[nn] == labels)[valid].mean())
        # 层次聚类（平均链接）阈值扫描
        np.fill_diagonal(D, 0)
        Z = linkage(squareform(D, checks=False), "complete" if name.endswith("约束") else "average")
        best = None
        rows = []
        for thr in np.linspace(0.05, 0.6, 23):
            cl = fcluster(Z, thr, "distance")
            k = len(set(cl))
            # 每组的多数球员；放错的段数
            wrong = 0
            for c in set(cl):
                lab = labels[cl == c]
                vals, cnt = np.unique(lab, return_counts=True)
                wrong += len(lab) - cnt.max()
            purity = 1 - wrong / len(items)
            actions = k + wrong
            rows.append((round(float(thr), 3), int(k), round(float(purity), 3), int(actions)))
            if best is None or actions < best[3]:
                best = rows[-1]
        report[name] = {"AUC": round(a, 3), "最近邻准确率": round(nn_acc, 3),
                        "最佳阈值": best[0], "组数": best[1], "纯度": best[2], "教练操作次数": best[3],
                        "不归组时的操作次数": len(items)}
        print(f"{name:10s} AUC={a:.3f} 最近邻={nn_acc:.3f} | 最佳阈值 {best[0]} → {best[1]} 组, 纯度 {best[2]:.2f}, 操作 {best[3]} 次（不归组需 {len(items)} 次）")
    json.dump(report, open(f"{evdir}/reid_report.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:4])
