"""Open a game's summary page through a small local server so your review is saved to a file.

    python serve.py ~/hoopstats/games/2026-10-12_vs_Tigers

The page is served from the game's hoopstats_output folder at http://127.0.0.1:<port>/summary.html
and every change you make is written to game_stats.json in the game folder (2 seconds after the
last change, and right away when you leave the page). Only this computer can reach the server.
Press Ctrl+C to stop it. `python run_game.py <folder> --serve` processes the game first, then does the same.
"""
import argparse
import datetime
import http.server
import json
import os
import re
import shutil
import socket
import sys
import threading
import webbrowser

OUT_DIR = "hoopstats_output"
STATS = "game_stats.json"
MAX_BODY = 20 * 1024 * 1024
LOCAL_HOSTS = ("127.0.0.1", "localhost")


def write_atomic(path, obj):
    """Write JSON to a temp file next to the target, then swap it in, so a crash never
    leaves a half-written file."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def backup_name(folder):
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    name = f"game_stats.{stamp}.json"
    n = 2
    while os.path.exists(os.path.join(folder, name)):
        name = f"game_stats.{stamp}_{n}.json"
        n += 1
    return name


def valid_payload(obj):
    """What the page sends: {"format": "hoopstats-game-stats", "data": {"clips": {...}, ...}}."""
    return (isinstance(obj, dict) and obj.get("format") == "hoopstats-game-stats"
            and isinstance(obj.get("data"), dict) and isinstance(obj["data"].get("clips", {}), dict))


def make_handler(game_folder):
    out = os.path.join(game_folder, OUT_DIR)
    stats = os.path.join(game_folder, STATS)
    lock = threading.Lock()

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=out, **kw)

        def log_message(self, fmt, *args):      # keep the terminal quiet except for saves
            pass

        # ---- guard: only pages from this server on this computer may call the API ----
        def local_request(self):
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
            if host not in LOCAL_HOSTS:
                return False
            origin = self.headers.get("Origin")
            if origin and not re.match(r"^http://(127\.0\.0\.1|localhost)(:\d+)?$", origin):
                return False
            return True

        def send_json(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def read_json(self):
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > MAX_BODY:
                raise ValueError("empty or too large")
            return json.loads(self.rfile.read(n).decode("utf-8"))

        def end_headers(self):
            if not self.path.startswith("/api/"):
                self.send_header("Cache-Control", "no-cache")
            super().end_headers()

        def do_GET(self):
            if self.path.split("?")[0] == "/api/state":
                if not self.local_request():
                    return self.send_json(403, {"error": "forbidden"})
                with lock:
                    if not os.path.exists(stats):
                        return self.send_json(200, {"exists": False, "file": STATS})
                    try:
                        with open(stats, encoding="utf-8") as f:
                            doc = json.load(f)
                    except Exception as e:
                        return self.send_json(500, {"error": f"{STATS} could not be read: {e}"})
                return self.send_json(200, {"exists": True, "file": STATS, "doc": doc,
                                            "mtime": os.path.getmtime(stats)})
            if self.headers.get("Range") and not self.path.startswith("/api/"):
                return self.send_range()
            return super().do_GET()

        def do_POST(self):
            path = self.path.split("?")[0]
            if path not in ("/api/state", "/api/import"):
                return self.send_json(404, {"error": "not found"})
            if not self.local_request():
                return self.send_json(403, {"error": "forbidden"})
            try:
                doc = self.read_json()
            except Exception as e:
                return self.send_json(400, {"error": f"bad request: {e}"})
            if not valid_payload(doc):
                return self.send_json(400, {"error": "not a HoopStats game_stats file"})
            with lock:
                backup = None
                if path == "/api/import" and os.path.exists(stats):
                    backup = backup_name(game_folder)
                    shutil.copy2(stats, os.path.join(game_folder, backup))
                try:
                    write_atomic(stats, doc)
                except Exception as e:
                    return self.send_json(500, {"error": f"could not write {STATS}: {e}"})
            if backup:
                print(f"Imported into {STATS}; the previous file was kept as {backup}", flush=True)
            return self.send_json(200, {"ok": True, "file": STATS, "backup": backup,
                                        "mtime": os.path.getmtime(stats)})

        # ---- byte ranges, so videos can be scrubbed (Safari needs this to play at all) ----
        def send_range(self):
            path = self.translate_path(self.path)
            if not os.path.isfile(path):
                return self.send_error(404)
            size = os.path.getsize(path)
            m = re.match(r"bytes=(\d*)-(\d*)$", self.headers.get("Range", "").strip())
            if not m or (m.group(1) == "" and m.group(2) == ""):
                return super().do_GET()
            if m.group(1) == "":
                start, end = max(0, size - int(m.group(2))), size - 1
            else:
                start = int(m.group(1))
                end = min(int(m.group(2)) if m.group(2) else size - 1, size - 1)
            if start >= size or start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Type", self.guess_type(path))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            with open(path, "rb") as f:
                f.seek(start)
                left = end - start + 1
                try:
                    while left > 0:
                        chunk = f.read(min(1 << 16, left))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        left -= len(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass          # the browser stops reading when you seek elsewhere

    return Handler


def free_port(start=8765):
    for port in range(start, start + 50):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise SystemExit("no free port between 8765 and 8814")


def serve(game_folder, open_browser=True, port=None):
    game_folder = os.path.abspath(os.path.expanduser(game_folder))
    page = os.path.join(game_folder, OUT_DIR, "summary.html")
    if not os.path.exists(page):
        raise SystemExit(f"{page} not found. Run: python run_game.py {game_folder}")
    port = port or free_port()
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), make_handler(game_folder))
    httpd.daemon_threads = True
    url = f"http://127.0.0.1:{port}/summary.html"
    print(f"Summary page: {url}")
    print(f"Your review is saved to {os.path.join(game_folder, STATS)}")
    print("Keep this window open while you review. Press Ctrl+C to stop.", flush=True)
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped. Your last changes were saved to", STATS)
    finally:
        httpd.server_close()


def main():
    ap = argparse.ArgumentParser(description="Open a game's summary page with saving to game_stats.json")
    ap.add_argument("folder", help="the game folder (the one holding hoopstats_output)")
    ap.add_argument("--port", type=int, default=None, help="port to use (default: first free one from 8765)")
    ap.add_argument("--no-browser", action="store_true", help="do not open the browser")
    a = ap.parse_args()
    serve(a.folder, not a.no_browser, a.port)


if __name__ == "__main__":
    sys.exit(main())
