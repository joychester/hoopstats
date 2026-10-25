"""Cut possession clips from one video, for review on the summary page (used by run_game.py).

The phone camera follows the ball. While it stays on one half, the team attacking that basket has
the ball; when it swings to the other half, the ball has just changed hands (defensive rebound,
steal, or an inbound after a score).

Clips are cut for both attack directions, so if the detected direction is wrong you can switch it
on the page without processing the video again.
"""
import os
import subprocess

import numpy as np



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
