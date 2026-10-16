"""把跟踪结果画到画面上：输出若干帧拼图或完整标注视频。"""
import cv2
import numpy as np


def color_for(i):
    rng = np.random.RandomState(i * 7 + 3)
    return tuple(int(v) for v in rng.randint(60, 255, 3))


def draw(img, items):
    """items: list of (box, label, color, thick)"""
    for b, label, col, th in items:
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, th)
        (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x1, y1 - tht - 8), (x1 + tw + 6, y1), col, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return img


def frame_items(tracks, k, label_fn, keep=lambda t: True):
    out = []
    for t in tracks:
        if not keep(t):
            continue
        for o in t.obs:
            if o[0] == k:
                lab, col, th = label_fn(t, o)
                out.append((o[1], lab, col, th))
    return out


def contact_sheet(video, detjson, tracks, ks, path, label_fn, keep=lambda t: True, cols=2, width=960):
    cap = cv2.VideoCapture(video)
    tiles = []
    for k in ks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, detjson["frames"][k]["frame"])
        ok, img = cap.read()
        draw(img, frame_items(tracks, k, label_fn, keep))
        cv2.putText(img, f"t={detjson['frames'][k]['t']:.1f}s", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        tiles.append(cv2.resize(img, (width, int(width * img.shape[0] / img.shape[1]))))
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(path, np.vstack(rows))
