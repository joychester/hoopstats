"""第 3 步：轨迹分类（本队 / 对手 / 裁判 / 场外）+ 断开轨迹拼接 + 输出复核数据。

用法: python analyze.py --video data/game1.mov --det out/game1_det.json --team black --out out/game1
"""
import argparse
import json
import os

import cv2
import numpy as np

from tracker import run


def track_stats(t, H):
    c = np.array([o[4] for o in t.obs if o[4]])
    w = np.array([o[3] for o in t.obs])
    b = np.array([o[1] for o in t.obs])
    return {
        "L": float(np.median(c[:, 0])) if len(c) else 0,
        "C": float(np.median(c[:, 1])) if len(c) else 0,
        "stripe": float(np.median(c[:, 2])) if len(c) else 0,
        "asp": float(np.median((b[:, 3] - b[:, 1]) / (b[:, 2] - b[:, 0]))),
        "y2": float(np.median(b[:, 3])),
        "spread": float(np.ptp(w[:, 0]) + np.ptp(w[:, 1])),
        "hair": float(np.median([o[4][3] for o in t.obs if o[4] and o[4][3] >= 0] or [0])),
    }


def classify(tracks, H, team, min_obs=8):
    """返回 {track_id: 'ours'|'opp'|'ref'|'off'|'short'}"""
    lab, cand = {}, []
    for t in tracks:
        s = t.stats = track_stats(t, H)
        if len(t.obs) < min_obs:
            lab[t.id] = "short"
        elif s["y2"] > H - 15:                       # 前景观众/替补席（贴底边）
            lab[t.id] = "off"
        elif s["asp"] < 2.1 and s["spread"] < 80:   # 看台上坐着不动的观众
            lab[t.id] = "off"
        elif s["C"] >= 12:                           # 本队球衣是黑/白（无彩色），彩色衣服都不是本队
            lab[t.id] = "opp" if s["y2"] else "off"
        elif s["stripe"] > 15:                       # 裁判条纹衫
            lab[t.id] = "ref"
        else:
            cand.append(t)
    # 在候选球员里按躯干亮度做 2 类聚类（加权：轨迹越长权重越大）
    Ls = np.array([t.stats["L"] for t in cand], np.float32)
    lo, hi = np.percentile(Ls, 20), np.percentile(Ls, 80)
    for _ in range(20):
        a = np.abs(Ls - lo) <= np.abs(Ls - hi)
        lo, hi = (Ls[a].mean() if a.any() else lo), (Ls[~a].mean() if (~a).any() else hi)
    thr = (lo + hi) / 2
    for t in cand:
        dark = t.stats["L"] < thr
        lab[t.id] = "ours" if (dark if team == "black" else not dark) else "opp"
    return lab, float(thr)


def stitch(tracks, fps, max_gap_s=4.0, speed_px_s=500):
    """把同一个人断开的轨迹接起来：后一段在前一段结束后 max_gap_s 内出现，
    场地坐标距离在可跑动范围内，躯干颜色相近。贪心按代价从小到大合并。"""
    tracks = sorted(tracks, key=lambda t: t.obs[0][0])
    pairs = []
    for a in tracks:
        if a.end_reason == "crowd":   # 因交叉/扎堆结束的轨迹不自动拼接
            continue
        ea, pa = a.obs[-1][0], np.array(a.obs[-1][3])
        for b in tracks:
            sb = b.obs[0][0]
            gap = (sb - ea) / fps
            if b is a or gap <= 0 or gap > max_gap_s:
                continue
            d = np.linalg.norm(np.array(b.obs[0][3]) - pa)
            if d > 120 + speed_px_s * gap:
                continue
            dh = abs(a.stats["hair"] - b.stats["hair"])
            if dh > 35:                  # 发色（金发 / 黑发）差太多，不是同一人
                continue
            dc = abs(a.stats["L"] - b.stats["L"]) / 40 + dh / 35
            pairs.append((d / 200 + gap / 2 + dc, a, b))
    pairs.sort(key=lambda x: x[0])
    nxt, prv = {}, {}
    for cost, a, b in pairs:
        if a.id in nxt or b.id in prv or cost > 3:
            continue
        # 防止成环 / 时间重叠：b 链上的所有段必须晚于 a 链
        nxt[a.id], prv[b.id] = b, a
    chains, seen = [], set()
    for t in tracks:
        if t.id in prv:
            continue
        chain, cur = [], t
        while cur is not None and cur.id not in seen:
            seen.add(cur.id)
            chain.append(cur)
            cur = nxt.get(cur.id)
        chains.append(chain)
    return chains


def best_crops(video, detjson, chain, n=4):
    """每条拼接轨迹挑 n 张最大、置信度最高的球员裁剪图。"""
    obs = [o for t in chain for o in t.obs]
    obs = sorted(obs, key=lambda o: -(o[1][3] - o[1][1]) * o[2])
    picked = []
    for o in obs:
        if all(abs(o[0] - p[0]) > 15 for p in picked):
            picked.append(o)
        if len(picked) == n:
            break
    cap = cv2.VideoCapture(video)
    crops = []
    for o in sorted(picked, key=lambda o: o[0]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, detjson["frames"][o[0]]["frame"])
        ok, img = cap.read()
        x1, y1, x2, y2 = map(int, o[1])
        pad = int(0.1 * (y2 - y1))
        c = img[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
        crops.append(cv2.resize(c, (int(160 * c.shape[1] / c.shape[0]), 160)))
    return crops


def pick_obs(obs_list, n):
    obs = sorted(obs_list, key=lambda o: -(o[1][3] - o[1][1]) * o[2])
    picked = []
    for o in obs:
        if all(abs(o[0] - p[0]) > 10 for p in picked):
            picked.append(o)
        if len(picked) == n:
            break
    return sorted(picked, key=lambda o: o[0])


def strip(crops, obs, h=160):
    """把选中的截图横向拼成一条。"""
    from grouping import box_key
    out = [crops[box_key(o[0], o[1])] for o in obs]
    out = [cv2.resize(c, (max(1, int(h * c.shape[1] / c.shape[0])), h)) for c in out if c.size]
    return np.hstack(out) if out else None


def infer_attack(tracks, labels):
    """自动判断本队进攻方向。依据：防守方总站在进攻球员和篮筐之间，
    所以无论镜头在哪个半场，本队整体都更靠近自己防守的篮筐。
    本队整体在对手左边 → 本队守左攻右。返回 (方向, 支持该方向的画面比例, 画面数)。"""
    pos = {"ours": {}, "opp": {}}
    for t in tracks:
        if labels.get(t.id) in pos:
            for o in t.obs:
                pos[labels[t.id]].setdefault(o[0], []).append(o[3][0])
    d = np.array([np.mean(pos["ours"][k]) - np.mean(pos["opp"][k]) for k in pos["ours"] if k in pos["opp"]])
    if len(d) < 30:
        return None, 0.0, int(len(d))
    left = float(np.mean(d < 0))
    return ("right" if left >= 0.5 else "left"), round(max(left, 1 - left), 3), int(len(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--det", required=True)
    ap.add_argument("--team", choices=["black", "white"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=None, help="grouping threshold; lower = more, purer groups")
    ap.add_argument("--no-groups", action="store_true",
                    help="skip appearance grouping (only needed for minutes); much faster")
    a = ap.parse_args()
    import shutil
    from grouping import THRESHOLD, build_chunks, chunk_samples, cluster, embed_chunks, read_crops
    shutil.rmtree(f"{a.out}/crops", ignore_errors=True)
    shutil.rmtree(f"{a.out}/review", ignore_errors=True)
    os.makedirs(f"{a.out}/crops", exist_ok=True)
    dj = json.load(open(a.det))
    fps = dj["fps"] / dj["step"]
    tracks = run(dj, a.video)
    labels, thr = classify(tracks, dj["height"], a.team)
    ours = sorted([t for t in tracks if labels[t.id] == "ours"], key=lambda t: t.obs[0][0])

    if a.no_groups:      # 只要进攻方向和球队分类（统计数据用），不做外观归组
        chunks, cl = [], []
    else:
        chunks = build_chunks(ours)
        crops = read_crops(a.video, dj, [o for c in chunks for o in chunk_samples(c)], pad=0.05)
        F = embed_chunks(chunks, crops)
        cl = cluster(F, chunks, a.threshold or THRESHOLD)

    # 组 -> 片段（同一轨迹里连续、同组的小块合成一个片段）
    groups = {}
    for c, g in zip(chunks, cl):
        pieces = groups.setdefault(int(g), [])
        if pieces and pieces[-1]["track"] == c["track"] and pieces[-1]["last_chunk_k1"] >= c["k0"] - 1:
            pieces[-1]["obs"] += c["obs"]
            pieces[-1]["last_chunk_k1"] = c["k1"]
        else:
            pieces.append({"track": c["track"], "obs": list(c["obs"]), "last_chunk_k1": c["k1"]})
    glist = sorted(groups.values(), key=lambda ps: -sum(len(p["obs"]) for p in ps))
    out = {"video": os.path.basename(a.video), "fps": dj["fps"], "step": dj["step"],
           "duration": dj["frames"][-1]["t"], "team": a.team, "L_threshold": thr,
           "counts": {k: sum(v == k for v in labels.values()) for k in set(labels.values())},
           "attack": dict(zip(("direction", "confidence", "frames"), infer_attack(tracks, labels))),
           "grouped": not a.no_groups, "groups": []}
    for pieces in glist:
        pieces.sort(key=lambda p: p["obs"][0][0])
    show = []
    for pieces in glist:
        show += pick_obs([o for p in pieces for o in p["obs"]], 6)
        for p in pieces:
            show += pick_obs(p["obs"], 3)
    shown = read_crops(a.video, dj, show)
    for gi, pieces in enumerate(glist, 1):
        gid = f"G{gi}"
        allobs = [o for p in pieces for o in p["obs"]]
        img = strip(shown, pick_obs(allobs, 6))
        if img is not None:
            cv2.imwrite(f"{a.out}/crops/{gid}.jpg", img)
        plist = []
        for pi, p in enumerate(pieces, 1):
            pid = f"{gid}-{pi}"
            pimg = strip(shown, pick_obs(p["obs"], 3), h=120)
            if pimg is not None:
                cv2.imwrite(f"{a.out}/crops/{pid}.jpg", pimg)
            plist.append({"id": pid, "track": p["track"],
                          "start": round(p["obs"][0][0] / fps, 2), "end": round(p["obs"][-1][0] / fps, 2),
                          "seen_s": round(len(p["obs"]) / fps, 1),
                          "boxes": {str(o[0]): o[1] for o in p["obs"]}})
        out["groups"].append({"id": gid, "pieces": plist, "seen_s": round(len(allobs) / fps, 1),
                              "start": plist[0]["start"], "end": plist[-1]["end"]})
    vis = np.zeros(len(dj["frames"]), int)
    for t in ours:
        for o in t.obs:
            vis[o[0]] += 1
    out["visible_ours"] = vis.tolist()
    out["all_tracks"] = {t.id: {"label": labels[t.id], **{k: round(v, 1) for k, v in t.stats.items()}}
                         for t in tracks}
    json.dump(out, open(f"{a.out}/tracklets.json", "w"))
    print(f"tracks={len(tracks)} ours={len(ours)} chunks={len(chunks)} -> {len(glist)} groups")
    for g in out["groups"]:
        print(f"  {g['id']:4s} {g['start']:5.1f}-{g['end']:5.1f}s seen={g['seen_s']:5.1f}s pieces={len(g['pieces'])}")


if __name__ == "__main__":
    main()
