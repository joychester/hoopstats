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


def segments(x, t, attack="right", min_len=2.0, detjson=None):
    """Split the video into stretches where the play is on the left or the right half.
    Returns [{"offense": side == attack, "side": "left"/"right", "start", "end"}].

    Normally this follows the camera (see camera_segments). When detjson is given, stretches
    where the camera did not follow the play are split by where the players are (see
    refine_with_players); everywhere else the camera result is kept exactly as it is."""
    out = camera_segments(x, t, min_len)
    if detjson is not None:
        out = refine_with_players(out, t, detjson, min_len)
    for g in out:
        g["offense"] = g["side"] == attack
    return out


def camera_segments(x, t, min_len=2.0):
    """Split by camera position into left / right half, with hysteresis so small wobbles do
    not count. Returns [] when the camera hardly turned at all."""
    xs = smooth(x, t)
    lo, hi = np.percentile(xs, 10), np.percentile(xs, 90)
    if hi - lo < 200:                                     # the camera barely turned: halves unknown
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
    return out


# ---- when the camera does not follow the play ----
LONG_SEG = 45.0        # a camera "possession" longer than this: the camera probably stopped following
MID_SEG = 25.0         # in such a video, camera stretches longer than this are checked too
MIN_PIECE = 4.0        # a player-based half change must hold this long to count
MIN_GAP = 300.0        # the two player clusters must be this far apart (first-frame pixels)
MIN_SEPARATION = 3.0   # the two groups must be this many times their spread apart (two clear groups)
MIN_SIZE_CORR = 0.3    # player size counts when it goes with position this strongly over the video
MIN_SIZE_RATIO = 1.35  # ... or players on one side look this much bigger than on the other


def player_signals(detjson):
    """Per processed frame, from the people detected (at least 3, else NaN):
    x of the median foot point in first-frame coordinates (camera turns taken out), and the
    median box height in pixels (players far from the camera look smaller)."""
    C = np.eye(3)
    xs, hs = [], []
    for f in detjson["frames"]:
        C = np.vstack([np.array(f["motion"]), [0, 0, 1]]) @ C
        Ci = np.linalg.inv(C)
        people = [d for d in f["dets"] if d[0] == 0 and d[1] > 0.4]
        if len(people) >= 3:
            xs.append(float(np.median([(Ci @ np.array([(d[2] + d[4]) / 2, d[5], 1]))[0] for d in people])))
            hs.append(float(np.median([d[5] - d[3] for d in people])))
        else:
            xs.append(np.nan)
            hs.append(np.nan)
    return np.array(xs), np.array(hs)


def two_groups(v):
    """Otsu split of 1-D values: the threshold that best separates two groups, and their medians."""
    vs = np.sort(v)
    best, thr = -1.0, vs[len(vs) // 2]
    for q in np.linspace(0.1, 0.9, 81):
        c = np.quantile(vs, q)
        lo, hi = vs[vs < c], vs[vs >= c]
        if len(lo) < 5 or len(hi) < 5:
            continue
        score = len(lo) * len(hi) * (hi.mean() - lo.mean()) ** 2
        if score > best:
            best, thr = score, c
    return thr, float(np.median(vs[vs < thr])), float(np.median(vs[vs >= thr]))


def refine_with_players(segs, t, detjson, min_len=2.0):
    """Split long camera stretches by where the players are.

    A camera that stays on one half for over LONG_SEG seconds (or never turns in a long video)
    has usually stopped following the ball: a phone left on a stand, or a wide shot of the
    whole court. There the players' median position still swings from half to half with each
    possession. Each such stretch is re-split with hysteresis around the line that best
    separates the two groups of player positions, and only when the players clearly change
    halves (pieces of at least MIN_PIECE seconds). Shorter camera stretches are left exactly
    as they were, so videos where the camera follows the play come out unchanged."""
    dur = float(t[-1])
    if segs:
        long_ones = [g for g in segs if g["end"] - g["start"] > LONG_SEG]
    else:
        long_ones = [{"side": None, "start": 0.0, "end": dur}] if dur > LONG_SEG else []
    if not long_ones:
        return segs
    px, ph = player_signals(detjson)
    ok = ~np.isnan(px)
    if ok.sum() < 20:
        return segs
    k = max(1, int(round(2.0 / (t[1] - t[0]))))         # 2-second average: players run, the medians jitter
    sm = lambda v: np.convolve(np.pad(np.interp(t, t[ok], v[ok]), (k // 2, k - 1 - k // 2), mode="edge"),
                               np.ones(k) / k, mode="valid")
    ps, far = sm(px), -np.log(sm(ph))                    # far: larger when the players look smaller
    # Unless the camera stands near the middle of the court, one half is nearer to it and the
    # players there look bigger. Which half that is follows from how size and position go
    # together over the whole video. Size does not drift the way the position estimate does.
    r = float(np.corrcoef(ps, far)[0, 1]) if ps.std() > 0 and far.std() > 0 else 0.0
    use_size, sign = abs(r) >= MIN_SIZE_CORR, (1.0 if r > 0 else -1.0)
    z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
    replace = {}
    queue, checked_more = list(long_ones), False
    while queue or (not checked_more and replace and segs):
        if not queue:
            # Once the camera has been shown not to follow in this video, shorter camera stretches
            # in it are checked as well: the person filming was likely doing the same there.
            checked_more = True
            queue = [h for h in segs if MID_SEG < h["end"] - h["start"] <= LONG_SEG]
            continue
        g = queue.pop(0)
        idx = np.flatnonzero((t >= g["start"]) & (t <= g["end"]))
        # Everything is measured inside this stretch only: the camera's position estimate drifts
        # over minutes, so one dividing line for the whole video does not fit every stretch.
        f = (z(ps[idx]) + sign * z(far[idx])) / 2 if use_size else z(ps[idx])
        thr, c_lo, c_hi = two_groups(f)
        low, high = f < thr, f >= thr
        spread = np.sqrt((f[low].std() ** 2 + f[high].std() ** 2) / 2) + 1e-9
        x_gap = np.median(ps[idx][high]) - np.median(ps[idx][low])
        size_ratio = np.exp(sign * (np.median(far[idx][high]) - np.median(far[idx][low])))
        clear = x_gap >= MIN_GAP or (use_size and size_ratio >= MIN_SIZE_RATIO)
        if os.environ.get("HOOPSTATS_DEBUG"):
            print(f"    stretch {g['start']:.0f}-{g['end']:.0f}s: separation {(c_hi - c_lo) / spread:.1f}, "
                  f"x gap {x_gap:.0f}, size ratio {size_ratio:.2f}, size used {use_size} (r={r:.2f})")
        if (c_hi - c_lo) / spread < MIN_SEPARATION or not clear:
            continue                                      # players stayed on one half: keep the camera's answer
        ps_idx = f
        band = 0.2 * (c_hi - c_lo)
        a, b = thr - band, thr + band
        state, sides = ps_idx[0] > thr, []
        for v in ps_idx:
            if state and v < a:
                state = False
            elif not state and v > b:
                state = True
            sides.append(state)
        pieces, s = [], 0
        for i in range(1, len(idx) + 1):
            if i == len(idx) or sides[i] != sides[s]:
                pieces.append({"side": "right" if sides[s] else "left", "start": float(t[idx[s]]),
                               "end": float(t[idx[i - 1]]), "by": "players"})
                s = i
        merged = []
        for p in pieces:                                 # short flips are noise: fold them into the neighbour
            if merged and (p["end"] - p["start"] < MIN_PIECE or merged[-1]["side"] == p["side"]):
                merged[-1]["end"] = p["end"]
            else:
                merged.append(p)
        if len(merged) > 1 and merged[0]["end"] - merged[0]["start"] < MIN_PIECE:
            merged[1]["start"] = merged[0]["start"]
            merged.pop(0)
        if len(merged) >= 2:
            replace[id(g)] = merged
    if not replace:
        return segs
    if not segs:
        return replace[id(long_ones[0])]
    out = []
    for g in segs:
        for p in replace.get(id(g), [g]):
            if out and out[-1]["side"] == p["side"]:
                out[-1]["end"] = p["end"]
            else:
                out.append(dict(p))
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


CLIPS_VERSION = 9
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
    segs = segments(x, t, detjson=detjson)
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
