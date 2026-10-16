"""导出某段视频本队原始轨迹（未拼接）的抽样截图，供人工标注标准答案。"""
import json, sys, os, pickle
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracker import run
from analyze import classify

video, det, team, out = sys.argv[1:5]
dj = json.load(open(det))
tracks = run(dj, video)
labels, _ = classify(tracks, dj["height"], team)
ours = [t for t in tracks if labels[t.id] == "ours"]
ours.sort(key=lambda t: t.obs[0][0])
os.makedirs(out, exist_ok=True)
pickle.dump({"ours": ours, "fps": dj["fps"] / dj["step"]}, open(f"{out}/tracks.pkl", "wb"))
cap = cv2.VideoCapture(video)
rows = []
for t in ours:
    o = t.obs
    pick = [o[int(i * (len(o) - 1) / 5)] for i in range(6)]
    tiles = []
    for ob in pick:
        cap.set(cv2.CAP_PROP_POS_FRAMES, dj["frames"][ob[0]]["frame"]); _, img = cap.read()
        x1, y1, x2, y2 = map(int, ob[1]); p = int(.12 * (y2 - y1))
        c = img[max(0, y1 - p):y2 + p, max(0, x1 - p):x2 + p]
        c = cv2.resize(c, (int(200 * c.shape[1] / c.shape[0]), 200))
        cv2.putText(c, f"{dj['frames'][ob[0]]['t']:.1f}", (2, 14), 0, .45, (0, 0, 255), 1)
        tiles.append(cv2.copyMakeBorder(c, 0, 0, 0, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    strip = np.hstack(tiles)
    row = np.full((210, 900, 3), 255, np.uint8)
    row[5:205, 110:110 + min(790, strip.shape[1])] = strip[:, :790]
    cv2.putText(row, f"#{t.id}", (5, 100), 0, .9, (0, 0, 0), 2)
    cv2.putText(row, f"n={len(o)}", (5, 130), 0, .5, (0, 0, 0), 1)
    rows.append(row)
for i in range(0, len(rows), 8):
    cv2.imwrite(f"{out}/sheet_{i // 8}.jpg", np.vstack(rows[i:i + 8]))
print(len(ours), "tracks:", [t.id for t in ours])
