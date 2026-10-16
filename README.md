# HoopStats: basketball stats from phone video (prototype)

Runs entirely on your Mac. Videos never leave your computer.

## What it does

- Finds every player, tracks them while the camera pans, and keeps only our team (black or white jersey). Referees, fans in the stands and adults on the sideline are filtered out.
- **Key clips review** (points, rebounds, assists, steals): the camera follows the ball, so when it swings to the half we attack, we have the ball. Each video is cut down to the moments we win the ball and each of our possessions. You watch those clips, pick what happened and who did it, and the stats add up. Turnovers are not counted for now.
- **Attack direction is detected automatically**: defenders stand between their man and the basket, so our team sits closer to the basket we defend. You can flip it on the page if it is wrong.
- **Minutes review**: groups clips of the same player by appearance (OSNet). You name each group and minutes are computed.

## Install (once)

Python 3.10+ and ffmpeg are required.

```bash
brew install python ffmpeg
cd ~/hoopstats
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install --only-binary=:all: -r requirements.txt
mkdir -p models
curl -L -o models/yolox_m.onnx https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.onnx
# OSNet (MIT licence): unzip and put osnet.onnx and osnet.data into models/
curl -L -o osnet.zip https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/osnet/releases/v0.62.2/osnet-onnx-float.zip
unzip osnet.zip && mv osnet-onnx-float/osnet.onnx osnet-onnx-float/osnet.data models/
```

## Process a game

1. Make one folder per game, e.g. `~/hoopstats/games/2026-10-12_vs_Tigers/`.
2. Put the game's videos in it (.MOV, .mp4, .m4v), plus a `player_info.txt`:

```
Jersey: black
7 Emma
11 Mia
23 Zoe
```

The first line is our jersey color (`black` or `white`; `球衣：黑` also works). Then one player per line, number first.

Optional but recommended: put the videos in two subfolders, `1st_half/` and `2nd_half/` (`H1`/`H2` or `上半场`/`下半场` also work). The attack direction is then decided separately for each half, so a switch of sides at half-time is never guessed. Without subfolders the program detects the switch on its own. Moving already-processed videos into half folders keeps their cached results.

3. Run:

```bash
cd ~/hoopstats
source .venv/bin/activate
python run_game.py ~/hoopstats/games/2026-10-12_vs_Tigers
```

4. Open `hoopstats_output/summary.html` inside the game folder.

Run it again after adding more videos: finished videos are skipped. A video that fails is skipped and the error goes to `hoopstats_output/run_log.txt`. Options: `--no-minutes` skips the minutes pages and appearance grouping (much faster); `--redo` reprocesses everything except detection; `--accurate` uses the larger YOLOX-M model at 15 fps (about 3x slower). By default detection uses YOLOX-S at 10 fps; download it with `curl -L -o models/yolox_s.onnx https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx` (without it, YOLOX-M is used). Press Ctrl+C once to stop; run the same command to continue.

Each possession ends as a made 2 or 3, free throws, a missed shot or a team turnover (by type: bad pass, out of bounds, offensive foul, ball handling, traveling, moving screen, technical foul, other). For every made or missed shot you can click where it was taken on a half-court diagram; the shot chart panel shows made and missed shots by zone (paint, mid-range, 3PT) with a heat map of made, missed or all shots, and exports as a PNG.

Your review is saved in the browser. Use **Save review** to keep a copy as a file and **Load review** to restore it.

## Pipeline

| Step | File | What it does |
| --- | --- | --- |
| 1 | `detect_video.py` | YOLOX person/ball detection and camera-motion estimate, cached as `detections.json` |
| 2 | `tracker.py` | Camera-compensated tracking with jersey-color gating |
| 3 | `analyze.py`, `grouping.py` | Team/referee/spectator classification, attack direction, OSNet appearance grouping |
| 4 | `events.py` | Cuts key clips for both attack directions |
| 5 | `export_review.py` | Minutes review page with annotated video |
| 6 | `run_game.py`, `game_template.html` | Batch run for one game folder and the summary page |

`run.py` still processes a single video. `eval/` holds the ground-truth labels and evaluation scripts.

## Known limits

- Our jersey numbers are too dark to read, so identities come from you.
- Players of the same team who cross can be swapped by the tracker; grouping recovers part of this.
- Key clips rely on the camera following the ball. A single video that spans half-time needs splitting.
- Off-camera players cannot be seen; minutes count a player on court for up to N seconds off camera (adjustable).
- Inference uses the CPU by default. Set `HOOPSTATS_COREML=1` to try Core ML.
