"""Fast frame reading through ffmpeg.

OpenCV's VideoCapture can be very slow on macOS, and the pipeline only needs every
Nth frame. ffmpeg decodes (with hardware help where available) and hands back just
the frames we ask for. Falls back to OpenCV when ffmpeg is not installed.
"""
import shutil
import subprocess

import cv2
import numpy as np


def video_info(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ok, img = cap.read()            # first frame gives the displayed (rotated) size
    cap.release()
    if not ok:
        return fps, 0, 0
    return fps, img.shape[1], img.shape[0]


def iter_frames(path, step=1, want=None):
    """Yield (frame_index, BGR image) for every `step`-th frame, or only frames in `want`."""
    fps, W, H = video_info(path)
    if W == 0:
        return
    want = set(want) if want is not None else None
    if want is not None and not want:
        return
    if shutil.which("ffmpeg"):
        if want is not None:
            expr = "+".join(f"eq(n\\,{n})" for n in sorted(want)) if len(want) < 400 else None
        else:
            expr = f"not(mod(n\\,{step}))" if step > 1 else None
        vf = [f"select='{expr}'"] if expr else []
        cmd = ["ffmpeg", "-v", "error", "-hwaccel", "auto", "-i", path]
        if vf:
            cmd += ["-vf", ",".join(vf), "-fps_mode", "passthrough"]
        cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        size = W * H * 3
        # frame numbers in output order
        if want is not None and expr:
            order = iter(sorted(want))
        elif want is None and step > 1:
            order = iter(range(0, 10 ** 9, step))
        else:
            order = iter(range(10 ** 9))
        try:
            for n in order:
                buf = p.stdout.read(size)
                if len(buf) < size:
                    break
                if want is not None and not expr and n not in want:
                    continue
                yield n, np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy()
        finally:
            p.stdout.close()
            p.kill()
            p.wait()
        return
    cap = cv2.VideoCapture(path)
    i, last = 0, (max(want) if want is not None else None)
    while True:
        if last is not None and i > last:
            break
        ok = cap.grab()
        if not ok:
            break
        if (want is not None and i in want) or (want is None and i % step == 0):
            ok, img = cap.retrieve()
            if ok:
                yield i, img
        i += 1
    cap.release()
