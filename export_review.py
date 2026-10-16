"""第 4 步：生成复核包（双击 index.html 即可打开，无需联网）。

产物: OUT/review/index.html, data.js, crops/T*.jpg, video.mp4（带本队轨迹编号的标注视频）
用法: python export_review.py --video data/game1.mov --analysis out/game1 --title "第一场"
"""
import argparse
import json
import os
import shutil
import subprocess

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))


def render_video(video, tl, path, width=960):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(3)), int(cap.get(4))
    s = width / W
    h = int(H * s) // 2 * 2
    step = tl["step"]
    # 处理帧序号 -> [(tracklet id, box)]
    by_k = {}
    for g in tl["groups"]:
        for p in g["pieces"]:
            for k, b in p["boxes"].items():
                by_k.setdefault(int(k), []).append((g["id"], b))
    ff = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{h}",
         "-r", str(fps), "-i", "-", "-i", video, "-map", "0:v", "-map", "1:a?", "-c:v", "libx264",
         "-preset", "veryfast", "-crf", "27", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "64k",
         "-shortest", "-movflags", "+faststart", path], stdin=subprocess.PIPE)
    i = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        img = cv2.resize(img, (width, h))
        for tid, b in by_k.get(round(i / step), []):
            x1, y1, x2, y2 = [int(v * s) for v in b]
            cv2.rectangle(img, (x1, y1), (x2, y2), (40, 200, 255), 2)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + 12 * len(tid) + 8, y1), (40, 200, 255), -1)
            cv2.putText(img, tid, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        ff.stdin.write(img.tobytes())
        i += 1
    ff.stdin.close()
    ff.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--title", default="Game review")
    ap.add_argument("--roster", default="", help="roster text to prefill (one player per line)")
    ap.add_argument("--back", default="", help="link back to the game summary page")
    a = ap.parse_args()
    tl = json.load(open(f"{a.analysis}/tracklets.json"))
    rv = f"{a.analysis}/review"
    os.makedirs(f"{rv}/crops", exist_ok=True)
    for f in os.listdir(f"{a.analysis}/crops"):
        shutil.copy(f"{a.analysis}/crops/{f}", f"{rv}/crops/{f}")
    render_video(a.video, tl, f"{rv}/video.mp4")
    data = {"title": a.title, "video": tl["video"], "duration": tl["duration"], "team": tl["team"],
            "roster": a.roster, "back": a.back,
            "groups": [{"id": g["id"], "start": g["start"], "end": g["end"], "seen_s": g["seen_s"],
                        "pieces": [{k: p[k] for k in ("id", "start", "end", "seen_s")} for p in g["pieces"]]}
                       for g in tl["groups"]]}
    with open(f"{rv}/data.js", "w") as f:
        f.write("window.GAME = " + json.dumps(data, ensure_ascii=False) + ";\n")
    shutil.copy(f"{HERE}/review_template.html", f"{rv}/index.html")
    print("review package:", rv)


if __name__ == "__main__":
    main()
