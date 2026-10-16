"""一键处理一段比赛视频，生成复核包。

用法:
    python run.py 视频文件.mov --team black --title "对XX队 第一节"

完成后打开 output/<视频名>/review/index.html
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--team", choices=["black", "white"], required=True, help="本队球衣颜色")
    ap.add_argument("--title", default=None)
    ap.add_argument("--model", default=os.path.join(HERE, "models", "yolox_m.onnx"))
    ap.add_argument("--out", default=os.path.join(HERE, "output"))
    ap.add_argument("--attack", choices=["left", "right"], default=None,
                    help="视频开始时本队进攻哪一侧篮筐；提供后额外生成“关键片段复核”页")
    a = ap.parse_args()
    name = os.path.splitext(os.path.basename(a.video))[0]
    out = os.path.join(a.out, name)
    os.makedirs(out, exist_ok=True)
    det = os.path.join(out, "detections.json")
    py = sys.executable
    if not os.path.exists(det):  # 检测结果已缓存则跳过（最耗时的一步）
        subprocess.check_call([py, os.path.join(HERE, "detect_video.py"), a.video, det, "--model", a.model])
    subprocess.check_call([py, os.path.join(HERE, "analyze.py"), "--video", a.video, "--det", det,
                           "--team", a.team, "--out", out])
    subprocess.check_call([py, os.path.join(HERE, "export_review.py"), "--video", a.video,
                           "--analysis", out, "--title", a.title or name])
    print("\n完成。上场时间复核：", os.path.join(out, "review", "index.html"))
    if a.attack:
        subprocess.check_call([py, os.path.join(HERE, "events.py"), "--video", a.video, "--det", det,
                               "--attack", a.attack, "--out", os.path.join(out, "events"),
                               "--title", (a.title or name) + "：关键片段"])
        print("关键片段复核（得分/篮板/助攻/抢断）：", os.path.join(out, "events", "index.html"))


if __name__ == "__main__":
    main()
