# set_pieces.py -- needs np, pd, FPS, HOME_IDS, AWAY_IDS, real_halves and animate_frames
# (paste below your plotting functions, or `from plotting import ...` after renaming the file).

import numpy as np
import pandas as pd
from Plotting import animate_frames
from Algorithm import sanity_check
FPS = 25               # frames per second (used to convert seconds to frames)

BALL_STOP = 0.1       # ball_speed below this = ball stopped (same unit as your TH_BALL_SPEED)
PLAYER_STOP = 1.0     # median player speed below this (m/s) = players standing
MIN_OUT_S, MIN_STOP_S, MIN_MISSING_S = 1, 10, 25   # seconds a condition must last to count
HOME_IDS = list(range(1, 25))   # Jugadores del 1 al 24
AWAY_IDS = list(range(25, 47)) 
BOX_DEPTH = 16.5                 # penalty area depth (m)
BOX_HALF_WIDTH = 20.16 

MATCH_META = {"offsets": {1: 1702057, 2: 1702058},
              "halves": {1: (1713242, 1792708), 2: (1812866, 1892255)},
              "kickoffs": {1: {"minute": 7, "second": 27, "frame": 11},
                           2: {"minute": 75, "second": 49, "frame": 16}}}
k = MATCH_META["kickoffs"]
real_halves = {h: ((k[h]["minute"] * 60 + k[h]["second"]) * 25 + k[h]["frame"],
                   MATCH_META["halves"][h][1] - MATCH_META["offsets"][h]) for h in [1, 2]}

def _long_runs(mask, min_len):
    """Keep only the runs of True that last at least min_len frames."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    out = np.zeros(len(mask), dtype=bool)
    for s, e in zip(edges[::2], edges[1::2]):
        if e - s >= min_len:
            out[s:e] = True
    return out


def detect_set_pieces(df, halves=None):
    """Per in-play frame: sp_reason in {'ball out', 'stopped', 'ball missing', None} and sp_flag."""
    halves = halves or real_halves
    speed_cols = [f"{i}_s" for i in HOME_IDS + AWAY_IDS]
    parts = []
    for s, e in halves.values():
        d = df[(df["frame_id"] >= s) & (df["frame_id"] <= e)]
        out = ((d["ball_x"] < 0) | (d["ball_x"] > 1) | (d["ball_y"] < 0) | (d["ball_y"] > 1)).to_numpy()
        stop = ((d["ball_speed"] < BALL_STOP) & (d[speed_cols].median(axis=1) < PLAYER_STOP)).to_numpy()
        miss = d["ball_x"].isna().to_numpy()
        reason = np.full(len(d), None, dtype=object)
        # assigned from lowest to highest priority: ball out > stopped > ball missing
        for name, m, secs in [("ball missing", miss, MIN_MISSING_S), ("stopped", stop, MIN_STOP_S),
                              ("ball out", out, MIN_OUT_S)]:
            reason[_long_runs(m, secs * FPS)] = name
        parts.append(pd.DataFrame({"frame_id": d["frame_id"].to_numpy(), "sp_reason": reason}))
    sp = pd.concat(parts, ignore_index=True)
    sp["sp_flag"] = sp["sp_reason"].notna()
    return sp


def mask_possession(chk, sp):
    """Add sp_flag / sp_reason to chk and remove possession (and box counts) during set pieces."""
    chk = chk.drop(columns=["sp_flag", "sp_reason"], errors="ignore").merge(sp, on="frame_id", how="left")
    m = chk["sp_flag"].fillna(False).astype(bool)
    chk.loc[m, ["possession_team", "att_in_box", "def_in_box"]] = np.nan
    return chk


def animate_frames_sp(df, start_frame, end_frame, chk, step=1, save_path=None,
                      video_path=None, video_offset=0, **kw):
    """animate_frames + a red 'SET PIECE (reason)' banner while sp_reason is set + tactical video on the right."""
    anim = animate_frames(df, start_frame, end_frame, chk=chk, step=step, **kw)   # saved below, after wrapping
    ids = df[(df["frame_id"] >= start_frame) & (df["frame_id"] <= end_frame)]["frame_id"].to_numpy()[::step]
    sp = chk.drop_duplicates("frame_id").set_index("frame_id")["sp_reason"].reindex(ids).to_numpy(dtype=object)

    fig = anim._fig
    ax = fig.axes[0]
    ax.invert_yaxis()
    banner = ax.text(0.5, 0.93, "", transform=ax.transAxes, ha="center", va="center", fontsize=16,
                     weight="bold", color="white", zorder=10, visible=False,
                     bbox=dict(boxstyle="round", facecolor="red", edgecolor="black"))

    if video_path:
        import imageio.v3 as iio
        meta = iio.immeta(video_path, plugin="pyav")
        vid_fps = meta.get("fps", FPS)
        if abs(vid_fps - FPS) > 0.5:
            print(f"warning: video fps = {vid_fps:.2f}, tracking fps = {FPS}")

        # Resize figure to 2 panels and move the pitch axes to the left half
        fig.set_size_inches(22, 8)
        ax.set_position([0.04, 0.12, 0.43, 0.80])
        vax = fig.add_axes([0.50, 0.08, 0.48, 0.84])
        vax.axis("off")
        vax.set_title("Tactical video")
        vim = vax.imshow(np.zeros((9, 16, 3), dtype=np.uint8))

    draw = anim._func

    def func(k):
        arts = draw(k)
        active = isinstance(sp[k], str)
        banner.set_visible(active)
        banner.set_text(f"SET PIECE ({sp[k]}) - no possession" if active else "")

        if video_path:
            target = int(ids[k]) + video_offset
            try:
                img = iio.imread(video_path, index=target, plugin="pyav")
                vim.set_data(img)
                vim.set_extent((0, img.shape[1], img.shape[0], 0))
            except Exception:
                pass
            return (*arts, banner, vim)

        return (*arts, banner)

    anim._func = func
    if save_path:
        anim.save(save_path, fps=FPS / step, dpi=110)
    return anim

def plot_match_timeline(chk, halves=None, ax=None):
    HOME_TAG, AWAY_TAG = "H", "A"
    HOME_COLOUR = "#d62728"
    AWAY_COLOUR = "#1f77b4"
    BALL_COLOUR = "white"
    PITCH_COLOUR = "#3a7d44"
    """Whole-match timeline: who is attacking (possession) and when a set piece is happening.
    x axis = playing time in minutes (half-time removed). chk must come from mask_possession()."""
    halves = halves or real_halves
    (s1, e1), (s2, _) = halves[1], halves[2]
    len1 = (e1 - s1) / FPS / 60                                  # length of the first half (min)
    f = chk["frame_id"].to_numpy()
    x = np.where(chk["half"].to_numpy() == 1, (f - s1) / FPS / 60, len1 + (f - s2) / FPS / 60)
    dt = 1 / FPS / 60
    poss, sp = chk["possession_team"], chk["sp_reason"]

    rows = [(f"{HOME_TAG} attacking", poss == HOME_TAG, HOME_COLOUR, 3, None),
            (f"{AWAY_TAG} attacking", poss == AWAY_TAG, AWAY_COLOUR, 2, None)]
    rows += [("Set piece", sp == r, c, 1, r) for r, c in
             [("ball out", "purple"), ("stopped", "orange"), ("ball missing", "grey")]]
    ax = ax or plt.subplots(figsize=(15, 3))[1]
    for name, m, colour, y, label in rows:
        m = m.to_numpy(dtype=bool)
        edges = np.flatnonzero(np.diff(np.r_[0, m.astype(np.int8), 0]))
        spans = [(x[s], x[e - 1] - x[s] + dt) for s, e in zip(edges[::2], edges[1::2])]
        ax.broken_barh(spans, (y - 0.4, 0.8), color=colour, label=label)
    ax.axvline(len1, color="black", ls="--", lw=1)
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(["Set piece", f"{AWAY_TAG} attacking", f"{HOME_TAG} attacking"])
    ax.set_xlim(0, x.max())
    ax.set_xlabel("playing time (min), dashed line = half-time")
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    ticks = np.arange(0, x.max(), 5)
    frames = [int(s1 + m * 60 * FPS) if m <= len1 else int(s2 + (m - len1) * 60 * FPS) for m in ticks]
    top = ax.twiny()
    top.set_xlim(ax.get_xlim())
    top.set_xticks(ticks)
    top.set_xticklabels(frames, fontsize=8)
    top.set_xlabel("frame_id")
    return ax

import matplotlib.pyplot as plt



df = pd.read_csv("data/dataframe_hackaton_final.csv", low_memory=False)
TEAM_MAP = {"Atletic Club": "A", "Barcelona": "H"}   # swap if the other mapping scored higher
df["players_in_the_box_team"] = df["players_in_the_box_team"].map(TEAM_MAP)
chk = sanity_check(df, halves=real_halves)
sp = detect_set_pieces(df)             # 1. finds the set-piece frames (sp_reason, sp_flag)
chk = mask_possession(chk, sp)         # 2. adds sp_reason to chk and blanks possession there
plot_match_timeline(chk); plt.show()
animate_frames_sp(
    df, 11185, 16185, chk=chk, step=5,
    video_path="C:/Users/nerir/OneDrive/Escritorio/exps_programacio/more-than-a-hack-2027/data/tactical/BAR-ATH_tactical.mp4",
    video_offset=0,
    save_path="comparison_neri.gif"
)
