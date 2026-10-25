# games/

Put each game in its own folder here. Nothing in this folder is committed except this README: game footage usually shows minors, so videos, rosters and results stay on your computer.

## What you create

```
games/
└── 2026-10-12_vs_Tigers/          one folder per game; any name works (date + opponent is handy)
    ├── player_info.txt            our jersey color and roster (see below)
    ├── 1st_half/                  videos filmed in the first half (.MOV, .mp4, .m4v)
    │   ├── IMG_3303.mp4
    │   └── IMG_3305.mp4
    └── 2nd_half/                  videos filmed in the second half
        ├── IMG_3338.mp4
        └── IMG_3339.mp4
```

`player_info.txt`: the first line is our jersey color, then one player per line, number first:

```
Jersey: black
7 Emma
11 Mia
23 Zoe
```

- The half folders are optional but recommended. `H1`/`H2` and `上半场`/`下半场` also work. With them, the attack direction is decided per half. Without them, put the videos straight into the game folder and the switch of sides is guessed.
- Use either half folders or videos directly in the game folder, not both.
- Videos are processed in file-name order, so keep the phone's names (IMG_0001, IMG_0002, ...).

## What HoopStats adds

```
games/2026-10-12_vs_Tigers/
├── game_stats.json                your review, saved automatically while you work on the summary page
├── game_stats.<date>_<time>.json  backups kept when you import a review over it
├── <game>_share_<date>_<time>.zip share zips made with "Download share zip"
└── hoopstats_output/
    ├── summary.html               the review page; run_game.py and serve.py open it for you
    ├── data.js                    data the summary page reads
    ├── run_log.txt                progress and errors of every run
    ├── share_cache/               640 px copies of confirmed clips, reused by the next share zip
    └── IMG_3303/                  one folder per video
        ├── detections.json        players and ball found in each frame, plus camera movement (the slow step, cached)
        ├── tracklets.json         players followed over time, team / referee / spectator labels, attack direction
        ├── clips.json             list of possession clips
        ├── clips/                 the clips themselves (C1.mp4, C2.mp4, ...) played on the summary page
        ├── crops/                 player snapshots used by the minutes page
        └── review/                minutes review page (only without --no-minutes)
```

Run the same command again after adding videos: videos that are already done are skipped. To start a video over, delete its folder in `hoopstats_output/`.

Your review (every result and player, shot spots, attack directions, the player list and game info) is saved to `game_stats.json` in the game folder, 2 seconds after each change. That works when the page was opened by `run_game.py` or `python serve.py <game folder>`; keep the Terminal window open while you review. Opening `summary.html` directly also works, but then the review stays in that browser only. Running `run_game.py` again, or deleting `hoopstats_output/`, never touches `game_stats.json`. To move a review to another computer, copy `game_stats.json` along with the game folder, or use **Export** / **Import game_stats.json** on the page.

```bash
cd ~/hoopstats && source .venv/bin/activate
python run_game.py ~/hoopstats/games/2026-10-12_vs_Tigers --no-minutes
```
