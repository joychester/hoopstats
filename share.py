"""Pack a game's review into a zip you can send to the team.

The zip holds the summary page, your review (game_stats data) and every confirmed clip,
re-encoded to 640 px wide so a whole game stays around 50-80 MB. Original videos, detection
files and logs are left out. Unzipped, summary.html opens offline in any browser as a
read-only shared copy.

Used by serve.py (the "Download share zip" button). Can also be run on its own:
    python share.py ~/hoopstats/games/2026-10-12_vs_Tigers
which packs what is in game_stats.json.
"""
import copy
import datetime
import json
import os
import re
import subprocess
import sys
import zipfile

OUT_DIR = "hoopstats_output"
STATS = "game_stats.json"
SHARE_WIDTH = 640
CACHE = "share_cache"

README = """HoopStats review: {title}
Shared copy made {made}.

Open summary.html (double-click it) in Chrome, Safari, Edge or Firefox on a computer.
Keep the clips folder next to it, or the videos will not play.

This is a read-only copy: you can browse, filter and play everything, and any changes
you make stay in your own browser only.

The clips show youth players. Please keep this file within the team and do not post it publicly.
"""


def load_game(out):
    """window.GAME_DATA from data.js."""
    text = open(os.path.join(out, "data.js"), encoding="utf-8").read()
    return json.loads(re.sub(r"^\s*window\.GAME_DATA\s*=\s*|;\s*$", "", text.strip()))


def confirmed_clips(game, data):
    """[(video, clip)] the page counts: confirmed clips on our attack side (or 'both'), plus the
    other team's clips that were confirmed as ours (marked "other")."""
    picked = []
    dirs, marks = data.get("dir") or {}, data.get("clips") or {}
    for v in game["videos"]:
        d = dirs.get(v["name"]) or v.get("attack") or "right"
        for c in v["clips"]:
            m = marks.get(v["name"] + "|" + c["id"]) or {}
            if m.get("done") and (c.get("side") in (d, "both") or m.get("other")):
                picked.append((v, c))
    return picked


def small_clip(out, v, c):
    """Re-encode one clip to SHARE_WIDTH wide, cached in hoopstats_output/share_cache."""
    src = os.path.join(out, c["src"])
    dst = os.path.join(out, CACHE, f"{v['dir']}_{c['id']}.mp4")
    if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-vf", f"scale={SHARE_WIDTH}:-2",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
                    "-c:a", "aac", "-b:a", "48k", "-movflags", "+faststart", tmp], check=True)
    os.replace(tmp, dst)
    return dst


def build_share(game_folder, data, progress=lambda done, total: None, template=None):
    """Write <game>_share_<time>.zip into the game folder. Returns its path."""
    out = os.path.join(game_folder, OUT_DIR)
    game = load_game(out)
    title = game["title"]
    picked = confirmed_clips(game, data)
    keep = {(v["name"], c["id"]) for v, c in picked}
    total = len(picked)
    progress(0, total)

    files = []                                    # (path in zip, source file)
    for n, (v, c) in enumerate(picked, 1):
        files.append((f"clips/{v['dir']}_{c['id']}.mp4", small_clip(out, v, c)))
        progress(n, total)

    shared = copy.deepcopy(game)                  # only confirmed clips, pointing at the small copies
    for v in shared["videos"]:
        v["clips"] = [dict(c, src=f"clips/{v['dir']}_{c['id']}.mp4") for c in v["clips"] if (v["name"], c["id"]) in keep]
        v["review"] = None
    made = datetime.datetime.now()
    review = {"exported": made.strftime("%Y-%m-%d %H:%M"), "data": data}
    page = open(template or os.path.join(out, "summary.html"), encoding="utf-8").read()
    page = page.replace('<script src="data.js"></script>', '<script src="data.js"></script>\n<script src="review.js"></script>', 1)

    root = f"{title}_shared/"
    name = f"{title}_share_{made:%Y-%m-%d_%H%M}.zip"
    path = os.path.join(game_folder, name)
    tmp = path + ".part"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(root + "summary.html", page)
        z.writestr(root + "data.js", "window.GAME_DATA = " + json.dumps(shared, ensure_ascii=False) + ";\n")
        z.writestr(root + "review.js", "window.SHARED_REVIEW = " + json.dumps(review, ensure_ascii=False) + ";\n")
        z.writestr(root + STATS, json.dumps({"format": "hoopstats-game-stats", "version": 1, "game": title,
                                             "saved": made.isoformat(), "data": data}, ensure_ascii=False, indent=1))
        z.writestr(root + "README.txt", README.format(title=title, made=made.strftime("%Y-%m-%d %H:%M")))
        for arc, src in files:                    # videos are already compressed: store them as they are
            z.write(src, root + arc, compress_type=zipfile.ZIP_STORED)
    os.replace(tmp, path)
    return path


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: python share.py <game folder>")
    folder = os.path.abspath(os.path.expanduser(sys.argv[1]))
    stats = os.path.join(folder, STATS)
    if not os.path.exists(stats):
        raise SystemExit(f"{stats} not found: review the game first (python serve.py {folder})")
    data = json.load(open(stats, encoding="utf-8"))["data"]

    def show(done, total):
        print(f"\rPacking clips {done} / {total}", end="", flush=True)
    path = build_share(folder, data, show)
    print(f"\nSaved {path} ({os.path.getsize(path) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
