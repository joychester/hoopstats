"""Process one game: every video in a folder + player_info.txt -> one summary review page.

Usage:
    python run_game.py ~/hoopstats/games/2026-10-12_vs_Tigers

Folder contents:
    player_info.txt   first line: our jersey color ("Jersey: black" or "Jersey: white"),
                      then one player per line ("7 Emma")
    *.MOV / *.mp4     the game's videos, processed in file-name order.
                      Optionally put them in two subfolders, 1st_half/ and 2nd_half/:
                      each half's attack direction is then decided on its own.

Results go to hoopstats_output/ inside the folder; open hoopstats_output/summary.html.
Videos already processed are skipped. A failing video does not stop the others;
errors are written to hoopstats_output/run_log.txt.
"""
import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO_EXT = (".mov", ".mp4", ".m4v")
OUT_DIR = "hoopstats_output"


class Log:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, *msg):
        line = f"[{datetime.datetime.now():%H:%M:%S}] " + " ".join(str(m) for m in msg)
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def parse_player_info(path):
    """Return (jersey color 'black'/'white', roster text)."""
    team, roster = None, []
    for raw in open(path, encoding="utf-8-sig"):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        low = line.lower()
        if team is None and re.search(r"球衣|颜色|jersey|team|color", low):
            if re.search(r"黑|black|dark|深", low):
                team = "black"
            elif re.search(r"白|white|light|浅", low):
                team = "white"
            continue
        roster.append(line)
    return team, "\n".join(roster)


def step(log, name, args):
    t0 = time.time()
    p = subprocess.Popen([sys.executable] + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    try:
        for line in p.stdout:          # show progress in the terminal (not in the log)
            lines.append(line)
            if line.startswith("  detecting"):
                print("    " + line.strip(), flush=True)
    except KeyboardInterrupt:          # Ctrl+C: stop the child process too
        p.terminate()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
        raise
    if p.wait() != 0:
        raise RuntimeError(f"{name} failed:\n{''.join(lines)[-3000:]}")
    detail = next((l.strip()[l.strip().find("("):] for l in lines if l.startswith("done:") and "(" in l), "")
    log(f"    {name} done in {time.time() - t0:.0f}s {detail}".rstrip())
    return "".join(lines)


def detect_args(accurate):
    """快速配置（默认）：小号模型 YOLOX-S、每 3 帧检测一次（30fps 视频即每秒 10 帧）。
    精细配置（--accurate）：YOLOX-M、每 2 帧检测一次。没有下载 YOLOX-S 时自动用 YOLOX-M。"""
    small = os.path.join(HERE, "models", "yolox_s.onnx")
    if accurate or not os.path.exists(small):
        return ["--model", os.path.join(HERE, "models", "yolox_m.onnx"), "--step", "2"]
    return ["--model", small, "--step", "3"]


def process_video(log, video, out, team, minutes, roster="", accurate=False):
    sys.path.insert(0, HERE)
    os.makedirs(out, exist_ok=True)
    det = os.path.join(out, "detections.json")
    if os.path.exists(det):
        log("    detections cached, skipping detection")
    else:
        step(log, "Detection", [os.path.join(HERE, "detect_video.py"), video, det + ".part"] + detect_args(accurate))
        os.replace(det + ".part", det)          # only a completed run counts as cache
    ana = os.path.join(out, "tracklets.json")
    if os.path.exists(ana) and minutes and not json.load(open(ana)).get("grouped", True):
        os.remove(ana)      # an earlier --no-minutes run skipped grouping; minutes need it now
    if not os.path.exists(ana):
        step(log, "Tracking & grouping", [os.path.join(HERE, "analyze.py"), "--video", video, "--det", det,
                            "--team", team, "--out", out] + ([] if minutes else ["--no-groups"]))
    tl = json.load(open(ana))
    if minutes and not os.path.exists(os.path.join(out, "review", "index.html")):
        step(log, "Minutes review page", [os.path.join(HERE, "export_review.py"), "--video", video,
                              "--analysis", out, "--title", os.path.basename(video),
                              "--roster", roster, "--back", "../../summary.html"])
    clips_json = os.path.join(out, "clips.json")
    from events import CLIPS_VERSION, cut_all
    clips = json.load(open(clips_json)) if os.path.exists(clips_json) else None
    if not clips or clips.get("version") != CLIPS_VERSION:     # new clip format: re-cut
        t0 = time.time()
        dj = json.load(open(det))
        name = os.path.basename(out)
        shutil.rmtree(os.path.join(out, "clips"), ignore_errors=True)
        allc, dur, segs = cut_all(video, dj, os.path.join(out, "clips"), rel_prefix=f"{name}/clips")
        clips = {"version": CLIPS_VERSION, "clips": allc, "duration": round(dur, 1), "segments": segs}
        json.dump(clips, open(clips_json, "w"), ensure_ascii=False)
        log(f"    cut {len(allc)} clips (both directions) in {time.time() - t0:.0f}s")
    att = tl.get("attack") or {}
    return {"name": os.path.basename(video), "dir": os.path.basename(out), "duration": clips["duration"],
            "attack_video": att.get("direction"), "confidence_video": att.get("confidence", 0),
            "frames": att.get("frames", 0),
            "clips": clips["clips"],
            "review": f"{os.path.basename(out)}/review/index.html" if minutes else None}


def half_of(name):
    """1 or 2 when a subfolder name marks a half ("1st_half", "H1", "first half", "上半场"), else None."""
    n = name.lower().replace("_", " ").replace("-", " ")
    if re.search(r"上半|first|1st|\bh1\b|\bhalf ?1\b|^1$", n):
        return 1
    if re.search(r"下半|second|2nd|\bh2\b|\bhalf ?2\b|^2$", n):
        return 2
    return None


def find_videos(folder):
    """Return [(half or None, path relative to folder)], or raise SystemExit with a message.

    Videos may sit directly in the game folder (halves are then detected automatically),
    or in two subfolders such as 1st_half/ and 2nd_half/ (each half is then judged on its own)."""
    is_video = lambda f: f.lower().endswith(VIDEO_EXT) and not f.startswith(".")
    top = sorted(f for f in os.listdir(folder) if is_video(f))
    halves = {}
    for d in sorted(os.listdir(folder)):
        p = os.path.join(folder, d)
        if d.startswith(".") or d == OUT_DIR or not os.path.isdir(p):
            continue
        h = half_of(d)
        vids = sorted(f for f in os.listdir(p) if is_video(f))
        if h and vids:
            if h in halves:
                raise SystemExit(f"two folders look like the same half: {halves[h][0]} and {d}")
            halves[h] = (d, vids)
    if not halves:
        return [(None, f) for f in top]
    if top:
        raise SystemExit("videos found both in the game folder and in half folders ("
                         + ", ".join(v[0] for v in halves.values())
                         + "). Move " + ", ".join(top) + " into the right half folder.")
    return [(h, os.path.join(d, f)) for h, (d, vids) in sorted(halves.items()) for f in vids]


def vote(results):
    """Frame-weighted vote. Returns (direction, share of frames agreeing, frames)."""
    w = n = 0
    for r in results:
        d, c, m = r["attack_video"], r["confidence_video"] or 0, r["frames"] or 0
        w += m * (c if d == "right" else (1 - c if d == "left" else 0.5))
        n += m
    s = w / n if n else 0.5
    return ("right" if s >= 0.5 else "left"), max(s, 1 - s), n


def game_direction(results, log):
    """Decide the attack direction for the whole game from every video together.

    Each video votes with its number of usable frames, so a 6-second clip cannot outvote
    a 2-minute one. When the videos are in half folders, each half is decided on its own.
    Otherwise a switch of sides (half-time) is accepted only when a run of
    consecutive videos at the start or end, at least 2 videos and 20% of all frames,
    clearly points the other way. Each video gets the game direction for its half."""
    if any(r.get("half") for r in results):
        for h in sorted({r["half"] for r in results}):
            part = [r for r in results if r["half"] == h]
            d, conf, _ = vote(part)
            for r in part:
                r["attack"], r["confidence"], r["attack_source"] = d, round(conf, 3), "half"
            log(f"{'1st' if h == 1 else '2nd'} half: we attack the {d} basket "
                f"({conf:.0%} of this half's frames agree, {len(part)} videos)")
        report_disagreements(results, log)
        return
    votes = []
    for r in results:
        d, c, n = r["attack_video"], r["confidence_video"] or 0, r["frames"] or 0
        left = c if d == "right" else (1 - c if d == "left" else 0.5)   # share of frames where we stand left
        votes.append((n * left, n))                                    # weighted "attack right" votes
    total = sum(n for _, n in votes) or 1

    def side(lo, hi):
        w = sum(v for v, _ in votes[lo:hi]); n = sum(m for _, m in votes[lo:hi]) or 1
        return ("right" if w / n >= 0.5 else "left"), max(w / n, 1 - w / n), n

    best = (None, side(0, len(votes)))
    for k in range(2, len(votes) - 1):
        a, b = side(0, k), side(k, len(votes))
        if a[0] != b[0] and min(a[2], b[2]) >= 0.2 * total and min(a[1], b[1]) >= 0.65 \
                and k >= 2 and len(votes) - k >= 2:
            if best[0] is None or min(a[1], b[1]) > min(best[1][1], best[2][1]):
                best = (k, a, b)
    if best[0] is None:
        d, conf, _ = best[1]
        for r in results:
            r["attack"], r["confidence"], r["attack_source"] = d, round(conf, 3), "game"
        log(f"Game direction: we attack the {d} basket ({conf:.0%} of all frames agree)")
    else:
        k, a, b = best
        for i, r in enumerate(results):
            d, conf = (a[0], a[1]) if i < k else (b[0], b[1])
            r["attack"], r["confidence"], r["attack_source"] = d, round(conf, 3), "game"
        log(f"Game direction: {a[0]} basket until {results[k - 1]['name']}, then {b[0]} (sides switched)")
    report_disagreements(results, log)


def report_disagreements(results, log):
    for r in results:
        if r["attack_video"] and r["attack_video"] != r["attack"]:
            log(f"    {r['name']}: own frames suggested {r['attack_video']} "
                f"({r['confidence_video']:.0%} of {r['frames']} frames); using {r['attack']}, decided from "
                f"{'its half' if r.get('attack_source') == 'half' else 'the whole game'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="folder holding one game")
    ap.add_argument("--no-minutes", action="store_true", help="skip the minutes review page (faster)")
    ap.add_argument("--redo", action="store_true", help="reprocess everything except detection")
    ap.add_argument("--accurate", action="store_true",
                    help="slower, finer detection (YOLOX-M at 15 fps); default is YOLOX-S at 10 fps")
    a = ap.parse_args()
    folder = os.path.abspath(os.path.expanduser(a.folder))
    out_root = os.path.join(folder, OUT_DIR)
    os.makedirs(out_root, exist_ok=True)
    log = Log(os.path.join(out_root, "run_log.txt"))
    log("=" * 60)
    log("Processing:", folder)

    info = os.path.join(folder, "player_info.txt")
    if not os.path.exists(info):
        log("ERROR: player_info.txt not found. First line: 'Jersey: black' or 'Jersey: white', then one player per line.")
        sys.exit(1)
    team, roster = parse_player_info(info)
    if team is None:
        log("ERROR: player_info.txt has no jersey color. Add 'Jersey: black' or 'Jersey: white' as the first line.")
        sys.exit(1)
    log(f"Our jersey: {team}; roster: {len(roster.splitlines())} players")

    try:
        found = find_videos(folder)
    except SystemExit as e:
        log(f"ERROR: {e}")
        sys.exit(1)
    if not found:
        log("ERROR: no videos found (.MOV .mp4 .m4v)")
        sys.exit(1)
    stems = [os.path.splitext(os.path.basename(v))[0] for _, v in found]
    if any(h for h, _ in found):
        for h in (1, 2):
            names = [os.path.basename(v) for hh, v in found if hh == h]
            if names:
                log(f"{'1st' if h == 1 else '2nd'} half: {len(names)} videos:", ", ".join(names))
    else:
        log(f"Found {len(found)} videos:", ", ".join(v for _, v in found))

    results, failed = [], []
    for i, (half, v) in enumerate(found, 1):
        stem = stems[i - 1]
        dup = stems.count(stem) > 1          # same file name in both halves
        out = os.path.join(out_root, f"h{half}_{stem}" if dup else stem)
        if a.redo:
            for f in ("tracklets.json", "clips.json"):
                if os.path.exists(os.path.join(out, f)):
                    os.remove(os.path.join(out, f))
            for d in ("review", "clips", "crops"):
                shutil.rmtree(os.path.join(out, d), ignore_errors=True)
        log(f"[{i}/{len(found)}] {v}")
        t0 = time.time()
        try:
            r = process_video(log, os.path.join(folder, v), out, team, not a.no_minutes, roster, a.accurate)
            r["half"] = half
            if dup:
                r["name"] = v
            results.append(r)
            d = r["attack_video"] or "undetermined"
            log(f"    done in {time.time() - t0:.0f}s; this video alone suggests the {d} basket "
                f"({r['confidence_video']:.0%} of {r['frames']} frames)")
        except KeyboardInterrupt:
            log("    Stopped by Ctrl+C. Run the same command again to continue; finished videos are kept.")
            sys.exit(130)
        except Exception as e:  # one failure does not stop the rest
            failed.append(v)
            log(f"    FAILED, skipped: {e}")
            log(traceback.format_exc())

    if results:
        game_direction(results, log)
    title = os.path.basename(folder)
    data = {"title": title, "team": team, "roster": roster, "videos": results, "failed": failed,
            "generated": f"{datetime.datetime.now():%Y-%m-%d %H:%M}"}
    with open(os.path.join(out_root, "data.js"), "w", encoding="utf-8") as f:
        f.write("window.GAME_DATA = " + json.dumps(data, ensure_ascii=False) + ";\n")
    shutil.copy(os.path.join(HERE, "game_template.html"), os.path.join(out_root, "summary.html"))
    log(f"Finished: {len(results)} succeeded, {len(failed)} failed")
    log("Open the summary page:", os.path.join(out_root, "summary.html"))


if __name__ == "__main__":
    main()
