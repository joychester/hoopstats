"""切出关键片段，供人工勾选得分、篮板、助攻、抢断。

原理：手机镜头跟着球走。镜头停在某个半场 = 进攻那个篮筐的一方在持球进攻；
镜头从一个半场转到另一个半场 = 球权刚换到进攻那一侧的一方（防守篮板 / 抢断 / 对方得分后发球）。

两个方向的片段都会切好：本队进攻右侧时，看“进攻右侧”的回合和“转向右侧”的时刻；反之亦然。
这样进攻方向判断错了，在页面上切换一下即可，不需要重新处理。

单段视频用法（旧）: python events.py --video V --det D.json --attack right --out OUT [--prefill P.json]
批量处理请用 run_game.py。
"""
import argparse
import json
import os
import shutil
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def camera_x(detjson):
    """每个处理帧：画面中心在第一帧坐标系里的横坐标（镜头左右转动的位置）。"""
    C = np.eye(3)
    xs = []
    W, H = detjson["width"], detjson["height"]
    for f in detjson["frames"]:
        C = np.vstack([np.array(f["motion"]), [0, 0, 1]]) @ C
        xs.append((np.linalg.inv(C) @ np.array([W / 2, H / 2, 1]))[0])
    return np.array(xs), np.array([f["t"] for f in detjson["frames"]])


def smooth(x, t):
    """1-second moving average, padded with the edge values (zero padding would drag the
    first and last samples toward 0 and invent a fake half at the start of the video)."""
    k = max(1, int(round(1.0 / (t[1] - t[0])))) if len(t) > 1 else 1
    xp = np.pad(x, (k // 2, k - 1 - k // 2), mode="edge")
    return np.convolve(xp, np.ones(k) / k, mode="valid")


def segments(x, t, attack="right", min_len=2.0):
    """把镜头位置分成左右两个半场（带滞回，避免来回抖动）。
    返回 [{"offense": 是否在 attack 一侧, "side": "left"/"right", "start", "end"}]。"""
    xs = smooth(x, t)
    lo, hi = np.percentile(xs, 10), np.percentile(xs, 90)
    if hi - lo < 200:                                     # 镜头几乎没转：无法判断半场
        return []
    a, b = lo + 0.35 * (hi - lo), lo + 0.65 * (hi - lo)
    right, state = [], (xs[0] > (a + b) / 2)
    for v in xs:
        if state and v < a:
            state = False
        elif not state and v > b:
            state = True
        right.append(state)
    right = np.array(right)
    segs, s = [], 0
    for i in range(1, len(right) + 1):
        if i == len(right) or right[i] != right[s]:
            segs.append({"side": "right" if right[s] else "left", "start": float(t[s]), "end": float(t[i - 1])})
            s = i
    out = []
    for g in segs:   # 合并太短的抖动段
        if out and (g["end"] - g["start"] < min_len) and out[-1]["side"] != g["side"]:
            out[-1]["end"] = g["end"]
            continue
        if out and out[-1]["side"] == g["side"]:
            out[-1]["end"] = g["end"]
        else:
            out.append(dict(g))
    tail = tail_segment(xs, t, out, a, b, lo, hi)
    if tail:
        out[-1]["end"] = float(round(tail["start"] - (t[1] - t[0]), 3))
        out.append(tail)
    for g in out:
        g["offense"] = g["side"] == attack
    return out


TAIL_MIN = 1.0         # seconds the camera must stay past the middle at the end of a video


def tail_segment(xs, t, segs, a, b, lo, hi):
    """A video often stops while the camera is still swinging to the other half (a fast
    break caught in the last seconds). The hysteresis then never fires and that possession
    is lost. If, at the very end, the camera has crossed the middle of the court toward
    the other half, stayed there TAIL_MIN seconds and travelled a good part of the court
    width, start a final segment for that half at the moment it crossed the middle."""
    if not segs:
        return None
    last = segs[-1]
    mid = (a + b) / 2
    to_right = last["side"] == "left"
    past = (xs > mid) if to_right else (xs < mid)
    if not past[-1]:
        return None
    i = len(xs) - 1
    while i > 0 and past[i - 1]:
        i -= 1
    if t[i] <= last["start"] or t[-1] - t[i] < TAIL_MIN:
        return None
    seg = xs[(t >= last["start"]) & (t <= t[i])]
    swing = (xs[-1] - seg.min()) if to_right else (seg.max() - xs[-1])
    if swing < 0.4 * (hi - lo):
        return None
    return {"side": "right" if to_right else "left", "start": float(t[i]), "end": float(t[-1]), "tail": True}


def cut(video, t0, t1, path):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t0:.2f}", "-i", video, "-t", f"{t1 - t0:.2f}",
                    "-vf", "scale=1280:-2", "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
                    "-c:a", "aac", "-b:a", "64k", "-movflags", "+faststart", path], check=True)


CLIPS_VERSION = 8
PRE_WIN = 7.0          # seconds kept before the camera swings to our half (5 s + 2 s extra lead-in)
MIN_FIRST = 8.0        # first segment of a video must last this long, or the ball is likely already the other team's


def cut_all(video, detjson, clip_dir, rel_prefix="clips"):
    """Cut one clip per possession, for both attack directions.

    Each clip starts 7 s before the camera swings to that half (so you see how the ball
    was won) and runs to the end of the possession. `side` is the basket being attacked;
    the page shows the clips whose side matches our attack direction.
    The camera lags the ball, so when a video starts and the camera switches halves within
    MIN_FIRST seconds, the other team usually already has the ball: that segment is skipped.
    Clip ids are numbered before skipping, so ids stay stable across versions.
    If the camera never moved between halves, the whole video becomes one clip (side
    "both") so nothing is lost."""
    x, t = camera_x(detjson)
    segs = segments(x, t)
    dur = float(t[-1])
    os.makedirs(clip_dir, exist_ok=True)
    clips = []
    for i, g in enumerate(segs):
        won = i > 0                                  # possession began inside this video
        skip = i == 0 and len(segs) > 1 and g["end"] - g["start"] < MIN_FIRST
        start = g["start"] - (PRE_WIN if won else 1.0)
        clips.append({"kind": "possession", "side": g["side"], "won": won,
                      "start": round(max(0.0, start), 1), "end": round(min(dur, g["end"] + 1.0), 1),
                      "skip": skip})
    if not clips:
        clips.append({"kind": "full", "side": "both", "won": False, "start": 0.0, "end": round(dur, 1)})
    clips.sort(key=lambda c: c["start"])
    for n, c in enumerate(clips, 1):
        c["id"] = f"C{n}"
    clips = [c for c in clips if not c.pop("skip", False)]
    for c in clips:
        cut(video, c["start"], c["end"], os.path.join(clip_dir, f"{c['id']}.mp4"))
        c["src"] = f"{rel_prefix}/{c['id']}.mp4"
    return clips, dur, segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--det", required=True)
    ap.add_argument("--attack", choices=["left", "right"], required=True, help="basket our team attacks at the start of the video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Key clips review")
    ap.add_argument("--prefill", default=None)
    a = ap.parse_args()
    dj = json.load(open(a.det))
    shutil.rmtree(a.out, ignore_errors=True)
    allclips, dur, _ = cut_all(a.video, dj, f"{a.out}/clips")
    clips = [c for c in allclips if c["side"] in (a.attack, "both")]   # single-video mode: our direction only
    for n, c in enumerate(clips, 1):                          # 重新编号，便于阅读
        new = f"C{n}"
        os.rename(f"{a.out}/{c['src']}", f"{a.out}/clips/{new}.tmp.mp4")
        c["id"], c["src"] = new, f"clips/{new}.mp4"
    for f in os.listdir(f"{a.out}/clips"):
        p = f"{a.out}/clips/{f}"
        if f.endswith(".tmp.mp4"):
            os.replace(p, p.replace(".tmp.mp4", ".mp4"))
        elif not any(f == f"{c['id']}.mp4" for c in clips):
            os.remove(p)
    pre = json.load(open(a.prefill)) if a.prefill else {}
    for c in clips:
        for p in pre.get("clips", []):
            if c["kind"] == p["kind"] and c["start"] - 1 <= p["at"] <= c["end"] + 1:
                c["prefill"] = p
    covered = sum(c["end"] - c["start"] for c in clips)
    data = {"title": a.title, "video": os.path.basename(a.video), "duration": round(dur, 1),
            "covered": round(covered, 1), "roster": pre.get("roster", ""), "clips": clips}
    with open(f"{a.out}/data.js", "w") as f:
        f.write("window.EVENTS = " + json.dumps(data, ensure_ascii=False) + ";\n")
    shutil.copy(os.path.join(HERE, "events_template.html"), f"{a.out}/index.html")
    print(f"{len(clips)} clips, {covered:.0f}s total (video {dur:.0f}s)")


if __name__ == "__main__":
    main()
