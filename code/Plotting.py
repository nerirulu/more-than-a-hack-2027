import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Arc
from matplotlib.lines import Line2D

# ==========================================
# CONFIGURATION
# ==========================================
PITCH_LENGTH = 105.0   # metres (coordinates in the df are normalised 0-1)
PITCH_WIDTH = 68.0
BOX_DEPTH = 16.5                 # penalty area depth (m)
BOX_HALF_WIDTH = 20.16
FPS = 25

HOME_TAG, AWAY_TAG = "H", "A"
HOME_IDS = list(range(1, 25))    # players 1-24
AWAY_IDS = list(range(25, 47))   # players 25-46

HOME_COLOUR = "#d62728"
AWAY_COLOUR = "#1f77b4"
BALL_COLOUR = "white"
PITCH_COLOUR = "#3a7d44"

# Velocity columns ({id}_x_vel, {id}_y_vel) are drawn as arrows.
# ARROW_SCALE = metres of arrow per unit of velocity. Tune it to the units of your *_vel columns.
ARROW_SCALE = 100.0
BALL_ARROW_SCALE = 100.0


# ==========================================
# PITCH
# ==========================================
def draw_pitch(ax=None, length=PITCH_LENGTH, width=PITCH_WIDTH):
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor(PITCH_COLOUR)
    line = dict(color="white", lw=1.5, zorder=1)

    ax.add_patch(Rectangle((0, 0), length, width, fill=False, **line))
    ax.plot([length / 2] * 2, [0, width], **line)
    ax.add_patch(Circle((length / 2, width / 2), 9.15, fill=False, **line))
    ax.scatter(length / 2, width / 2, s=10, color="white", zorder=1)

    for x0, sign in [(0, 1), (length, -1)]:
        # penalty area (16.5 x 40.32), goal area (5.5 x 18.32)
        ax.add_patch(Rectangle((x0 if sign == 1 else x0 - 16.5, width / 2 - 20.16),
                               16.5, 40.32, fill=False, **line))
        ax.add_patch(Rectangle((x0 if sign == 1 else x0 - 5.5, width / 2 - 9.16),
                               5.5, 18.32, fill=False, **line))
        ax.scatter(x0 + sign * 11, width / 2, s=10, color="white", zorder=1)
        ax.add_patch(Arc((x0 + sign * 11, width / 2), 18.3, 18.3,
                         theta1=-53 if sign == 1 else 127,
                         theta2=53 if sign == 1 else 233, **line))

    ax.set_xlim(-3, length + 3)
    ax.set_ylim(-3, width + 3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


# ==========================================
# HELPERS
# ==========================================
def _get_frame_row(df, frame_id):
    """Return the row for frame_id (or the first available frame after it)."""
    mask = df["frame_id"] == frame_id
    if mask.any():
        return df.loc[mask].iloc[0]
    later = df.loc[df["frame_id"] >= frame_id]
    if later.empty:
        raise ValueError(f"frame_id {frame_id} not found")
    return later.iloc[0]


def _fmt_timestamp(ts):
    """'0 days 00:07:27.440000' -> '00:07:27.44' (works for Timedelta or str)."""
    try:
        ts = pd.Timedelta(ts)
        total = ts.total_seconds()
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h):02d}:{int(m):02d}:{s:05.2f}"
    except Exception:
        return str(ts)


def _players_in_frame(row):
    """List of (player_id, team_tag, x, y, vx, vy) for every tracked player (metres)."""
    players = []
    for pid in HOME_IDS + AWAY_IDS:
        x, y = row.get(f"{pid}_x", np.nan), row.get(f"{pid}_y", np.nan)
        if pd.isna(x) or pd.isna(y):
            continue
        vx = row.get(f"{pid}_x_vel", np.nan)
        vy = row.get(f"{pid}_y_vel", np.nan)
        team = HOME_TAG if pid in HOME_IDS else AWAY_TAG
        players.append((pid, team, x * PITCH_LENGTH, y * PITCH_WIDTH,
                        0.0 if pd.isna(vx) else vx, 0.0 if pd.isna(vy) else vy))
    return players


# ==========================================
# MAIN PLOT
# ==========================================
def plot_frame(df, frame_id, ax=None, show_ids=True, show_arrows=True):
    """Draw one tracking frame.

    - All tracked players (red = home, blue = away) and the ball, with velocity arrows.
    - A line joining the ball to the nearest player.
    - Legend with: timestamp, team in possession (nearest-player proxy),
      set piece (if the 'set_pieces' column has a value) and distance to the nearest player.
    """
    ax = draw_pitch(ax)
    row = _get_frame_row(df, frame_id)
    players = _players_in_frame(row)

    # ---- players
    for pid, team, x, y, vx, vy in players:
        colour = HOME_COLOUR if team == HOME_TAG else AWAY_COLOUR
        ax.scatter(x, y, s=130, color=colour, edgecolors="black", linewidths=0.6, zorder=3)
        if show_ids:
            ax.text(x, y, str(pid), ha="center", va="center", fontsize=6.5,
                    color="white", weight="bold", zorder=5)
        if show_arrows and (vx or vy):
            ax.annotate("", xy=(x + vx * ARROW_SCALE, y + vy * ARROW_SCALE), xytext=(x, y),
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.2), zorder=4)

    # ---- ball
    bx, by = row.get("ball_x", np.nan), row.get("ball_y", np.nan)
    has_ball = pd.notna(bx) and pd.notna(by)
    nearest, dist = None, np.nan
    if has_ball:
        bx_m, by_m = bx * PITCH_LENGTH, by * PITCH_WIDTH
        ax.scatter(bx_m, by_m, s=90, color=BALL_COLOUR, edgecolors="black",
                   linewidths=1.3, zorder=6)
        bvx, bvy = row.get("ball_x_vel", np.nan), row.get("ball_y_vel", np.nan)
        if show_arrows and pd.notna(bvx) and pd.notna(bvy) and (bvx or bvy):
            ax.annotate("", xy=(bx_m + bvx * BALL_ARROW_SCALE, by_m + bvy * BALL_ARROW_SCALE),
                        xytext=(bx_m, by_m),
                        arrowprops=dict(arrowstyle="->", color="yellow", lw=2), zorder=6)

        # ---- nearest player + connecting line
        if players:
            dists = [np.hypot(p[2] - bx_m, p[3] - by_m) for p in players]
            i = int(np.argmin(dists))
            nearest, dist = players[i], dists[i]
            ax.plot([bx_m, nearest[2]], [by_m, nearest[3]], color="white",
                    ls="--", lw=1.3, zorder=2)
            ax.scatter(nearest[2], nearest[3], s=260, facecolors="none",
                       edgecolors="yellow", linewidths=2, zorder=4)

    # ---- legend info
    set_piece = row.get("set_pieces", np.nan)
    set_piece_txt = str(set_piece) if pd.notna(set_piece) else "-"
    if nearest is not None:
        possession_txt = f"{nearest[1]} (player {nearest[0]})"
        dist_txt = f"{dist:.1f} m"
    else:
        possession_txt, dist_txt = "unknown (ball not tracked)", "-"

    def info(text):
        return Line2D([], [], linestyle="none", label=text)

    handles = [
        Line2D([], [], marker="o", ls="none", color=HOME_COLOUR, mec="black", label=f"Home ({HOME_TAG})"),
        Line2D([], [], marker="o", ls="none", color=AWAY_COLOUR, mec="black", label=f"Away ({AWAY_TAG})"),
        Line2D([], [], marker="o", ls="none", color=BALL_COLOUR, mec="black", label="Ball"),
        info(f"Timestamp: {_fmt_timestamp(row.get('timestamp', ''))}"),
        info(f"Possession: {possession_txt}"),
        info(f"Set piece: {set_piece_txt}"),
        info(f"Ball-nearest player dist: {dist_txt}"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=4, frameon=True, fontsize=9)

    ax.set_title(f"frame {int(row['frame_id'])} | period {int(row.get('period_id', 0))}"
                 + ("" if has_ball else "   -- ball not tracked"))
    return ax


def save_frames(df, frame_ids, out_dir="frames"):
    """Save a PNG per frame (handy to build a video afterwards with ffmpeg)."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    for fid in frame_ids:
        fig, ax = plt.subplots(figsize=(12, 8))
        plot_frame(df, fid, ax=ax)
        fig.savefig(os.path.join(out_dir, f"frame_{fid:07d}.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)

# Paste this below plot_frame() in your plotting file.
# It reuses: draw_pitch, _fmt_timestamp and the constants (PITCH_LENGTH, PITCH_WIDTH, FPS,
# HOME_IDS, AWAY_IDS, HOME_TAG, AWAY_TAG, *_COLOUR, ARROW_SCALE, BALL_ARROW_SCALE).

# Replace your animate_frames() in the plotting file with this one.
# It reuses: draw_pitch, _fmt_timestamp and the constants (PITCH_LENGTH, PITCH_WIDTH, FPS,
# HOME_IDS, AWAY_IDS, HOME_TAG, AWAY_TAG, *_COLOUR, ARROW_SCALE, BALL_ARROW_SCALE,
# BOX_DEPTH, BOX_HALF_WIDTH).
# Pass chk (output of sanity_check, with the 'home_attack_dir' column) to also show possession
# and how many players attack / defend in the box. Without chk it only shows detected players.

def animate_frames(df, start_frame, end_frame, chk=None, step=1, save_path=None, close_fig=False):
    BOX_DEPTH = 16.5                 # penalty area depth (m)
    BOX_HALF_WIDTH = 20.16 
    """Animate a span of tracking frames.

    Info box under the pitch:
        line 1: timestamp | set piece | nearest player to the ball
        line 2: detected players per team (non-missing tracks in that frame)
        line 3 (needs chk): possession (10 s rule) | attackers vs defenders inside the attacked box

    Notebook:
        from IPython.display import HTML
        anim = animate_frames(df, 11185, 11435, chk=chk, step=2, close_fig=True)
        HTML(anim.to_jshtml())
    Script:
        anim = animate_frames(df, 11185, 11435, chk=chk, step=2, save_path="clip.mp4")
    """
    from matplotlib.animation import FuncAnimation

    L, W = PITCH_LENGTH, PITCH_WIDTH
    ids = HOME_IDS + AWAY_IDS
    n_home = len(HOME_IDS)

    sub = df[(df["frame_id"] >= start_frame) & (df["frame_id"] <= end_frame)].iloc[::step]
    sub = sub.reset_index(drop=True)
    n_frames = len(sub)
    if n_frames == 0:
        raise ValueError("No frames between start_frame and end_frame")

    def arr(cols):
        return sub.reindex(columns=cols).to_numpy(dtype=float)

    PX = arr([f"{i}_x" for i in ids]) * L
    PY = arr([f"{i}_y" for i in ids]) * W
    PVX = np.nan_to_num(arr([f"{i}_x_vel" for i in ids])) * ARROW_SCALE
    PVY = np.nan_to_num(arr([f"{i}_y_vel" for i in ids])) * ARROW_SCALE
    BX = arr(["ball_x"])[:, 0] * L
    BY = arr(["ball_y"])[:, 0] * W
    BVX = np.nan_to_num(arr(["ball_x_vel"])[:, 0]) * BALL_ARROW_SCALE
    BVY = np.nan_to_num(arr(["ball_y_vel"])[:, 0]) * BALL_ARROW_SCALE
    timestamps = sub["timestamp"].to_numpy() if "timestamp" in sub else [""] * n_frames
    set_pieces = sub["set_pieces"].to_numpy() if "set_pieces" in sub else [np.nan] * n_frames
    frame_ids = sub["frame_id"].to_numpy()

    # detected players per team, per frame
    det_home = np.isfinite(PX[:, :n_home]).sum(axis=1)
    det_away = np.isfinite(PX[:, n_home:]).sum(axis=1)

    # sanity-check info aligned to these frames
    if chk is not None:
        needed = {"possession_team", "att_in_box", "def_in_box", "home_attack_dir"}
        if not needed.issubset(chk.columns):
            raise KeyError(f"chk is missing columns: {needed - set(chk.columns)}")
        c = chk.drop_duplicates("frame_id").set_index("frame_id").reindex(frame_ids)
        poss_arr = c["possession_team"].to_numpy(dtype=object)
        att_arr = c["att_in_box"].to_numpy(dtype=float)
        def_arr = c["def_in_box"].to_numpy(dtype=float)
        dir_arr = c["home_attack_dir"].to_numpy(dtype=float)

    # ---- artists
    fig, ax = plt.subplots(figsize=(12, 8))
    draw_pitch(ax)
    colours = [HOME_COLOUR] * n_home + [AWAY_COLOUR] * (len(ids) - n_home)

    players = ax.scatter(PX[0], PY[0], s=130, c=colours, edgecolors="black", linewidths=0.6, zorder=3)
    quiv = ax.quiver(PX[0], PY[0], PVX[0], PVY[0], angles="xy", scale_units="xy", scale=1,
                     width=0.002, color="black", zorder=4)
    labels = [ax.text(0, 0, str(pid), ha="center", va="center", fontsize=6.5,
                      color="white", weight="bold", zorder=5) for pid in ids]
    ball = ax.scatter([], [], s=90, color=BALL_COLOUR, edgecolors="black", linewidths=1.3, zorder=6)
    ball_arrow = ax.quiver([0], [0], [0], [0], angles="xy", scale_units="xy", scale=1,
                           width=0.004, color="yellow", zorder=6)
    near_line, = ax.plot([], [], color="white", ls="--", lw=1.3, zorder=2)
    near_ring = ax.scatter([], [], s=260, facecolors="none", edgecolors="yellow",
                           linewidths=2, zorder=4)
    box = Rectangle((0, W / 2 - BOX_HALF_WIDTH), BOX_DEPTH, 2 * BOX_HALF_WIDTH, facecolor="yellow",
                    alpha=0.25, edgecolor="yellow", lw=2, zorder=1, visible=False)
    border = Rectangle((0, 0), L, W, fill=False, lw=5, zorder=7, visible=False)
    att_label = ax.text(1.015, 0.5, "", transform=ax.transAxes, ha="left", va="center",
                    fontsize=12, weight="bold", color="white",
                    bbox=dict(boxstyle="round", facecolor="grey", edgecolor="black"))
    att_label.set_visible(chk is not None)      # only shown when chk is given
    fig.subplots_adjust(right=0.88)
    ax.add_patch(box)
    ax.add_patch(border)
    title = ax.set_title("")
    info = ax.text(0.5, -0.03, "", transform=ax.transAxes, ha="center", va="top", fontsize=10,
                   bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    handles = [
        Line2D([], [], marker="o", ls="none", color=HOME_COLOUR, mec="black", label=f"Home ({HOME_TAG})"),
        Line2D([], [], marker="o", ls="none", color=AWAY_COLOUR, mec="black", label=f"Away ({AWAY_TAG})"),
        Line2D([], [], marker="o", ls="none", color=BALL_COLOUR, mec="black", label="Ball"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9)

    def update(k):
        pos = np.column_stack([PX[k], PY[k]])
        players.set_offsets(pos)
        quiv.set_offsets(pos)
        quiv.set_UVC(PVX[k], PVY[k])
        for t, (x, y) in zip(labels, pos):
            vis = not (np.isnan(x) or np.isnan(y))
            t.set_visible(vis)
            if vis:
                t.set_position((x, y))

        # ---- ball and nearest player
        has_ball = not (np.isnan(BX[k]) or np.isnan(BY[k]))
        near_txt = "-"
        if has_ball:
            ball.set_offsets([[BX[k], BY[k]]])
            ball_arrow.set_offsets([[BX[k], BY[k]]])
            ball_arrow.set_UVC([BVX[k]], [BVY[k]])
            d = np.hypot(PX[k] - BX[k], PY[k] - BY[k])
            d = np.where(np.isnan(d), np.inf, d)
            j = int(np.argmin(d))
            if np.isfinite(d[j]):
                near_line.set_data([BX[k], PX[k, j]], [BY[k], PY[k, j]])
                near_ring.set_offsets([[PX[k, j], PY[k, j]]])
                near_txt = f"player {ids[j]} ({HOME_TAG if j < n_home else AWAY_TAG}, {d[j]:.1f} m)"
            else:
                near_line.set_data([], [])
                near_ring.set_offsets(np.empty((0, 2)))
        else:
            ball.set_offsets(np.empty((0, 2)))
            ball_arrow.set_UVC([0], [0])
            near_line.set_data([], [])
            near_ring.set_offsets(np.empty((0, 2)))

        # ---- possession, attackers / defenders (needs chk)
        line3 = ""
        box.set_visible(False)
        border.set_visible(False)
        att_label.set_text("ATTACKING\n?")
        att_label.get_bbox_patch().set_facecolor("grey")
        if chk is not None:
            poss = poss_arr[k]
            if poss in (HOME_TAG, AWAY_TAG):
                dh = int(dir_arr[k])
                side = dh if poss == HOME_TAG else -dh
                att_label.set_text(f"ATTACKING\n{poss}\n{'→' if side == 1 else '←'}")
                att_label.get_bbox_patch().set_facecolor(HOME_COLOUR if poss == HOME_TAG else AWAY_COLOUR)
                box.set_xy((L - BOX_DEPTH if side == 1 else 0, W / 2 - BOX_HALF_WIDTH))
                box.set_visible(True)
                border.set_edgecolor(HOME_COLOUR if poss == HOME_TAG else AWAY_COLOUR)
                border.set_visible(True)
                other = AWAY_TAG if poss == HOME_TAG else HOME_TAG
                line3 = (f"\nPossession: {poss}   |   In the attacked box: "
                         f"{poss} attacking = {att_arr[k]:.0f}, {other} defending = {def_arr[k]:.0f}")
            else:
                line3 = "\nPossession: unknown   |   In the attacked box: -"

        sp = set_pieces[k]
        sp_txt = str(sp) if pd.notna(sp) else "-"
        title.set_text(f"frame {int(frame_ids[k])}" + ("" if has_ball else "   -- ball not tracked"))
        info.set_text(f"Timestamp: {_fmt_timestamp(timestamps[k])}   |   Set piece: {sp_txt}   |   "
                      f"Nearest: {near_txt}\n"
                      f"Detected players:  {HOME_TAG} = {det_home[k]}   {AWAY_TAG} = {det_away[k]}"
                      f"{line3}")
        return (players, quiv, ball, ball_arrow, near_line, near_ring, box, border,
                title, info, *labels)

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / FPS * step, blit=False)

    if save_path:
        anim.save(save_path, fps=FPS / step, dpi=110)
    if close_fig:
        plt.close(fig)
    return anim

def minute_to_frame(df, minute, period=None):
    """frame_id whose timestamp is closest to `minute` (e.g. 12.5 = 12:30)."""
    d = df if period is None else df[df["period_id"] == period]
    return int(d.loc[(d["time_min"] - minute).abs().idxmin(), "frame_id"])

def plot_minute(df, minute, period=None, **kw):
    return plot_frame(df, minute_to_frame(df, minute, period), **kw)

def animate_minutes(df, start_min, end_min, period=None, **kw):
    return animate_frames(df,
                          minute_to_frame(df, start_min, period),
                          minute_to_frame(df, end_min, period), **kw)

from IPython.display import HTML
from Algorithm import sanity_check, summarize_sanity, _infer_attack_dir, _box_mask

def _flag(v):
    return "n/a (no label)" if pd.isna(v) else ("OK" if bool(v) else "MISMATCH")


def plot_frame_check(df, chk, frame_id, ax=None, attack_dir=None, show_ids=True):
    """plot_frame + everything sanity_check() computed for that frame.

    Extra elements:
      - coloured pitch border = team in possession according to the 10 s state machine
      - shaded yellow box = penalty area attacked by the possession team
      - green rings = players (both teams) currently inside that box
      - legend = nearest player, possession, recomputed att/def in box vs the labels, matches
    """
    L, W = PITCH_LENGTH, PITCH_WIDTH
    ax = draw_pitch(ax)
    row = _get_frame_row(df, frame_id)
    fid = int(row["frame_id"])
    sel = chk.loc[chk["frame_id"] == fid]
    c = sel.iloc[0] if len(sel) else None          # None -> frame outside the in-play ranges
    poss = c["possession_team"] if c is not None else None

    # ---- attacked box of the possession team
    side = None
    if poss in (HOME_TAG, AWAY_TAG):
        if attack_dir is not None:
            dirs = attack_dir[int(c["half"])]
        else:
            half_start = chk.loc[chk["half"] == c["half"], "frame_id"].min()
            dirs = _infer_attack_dir(df, half_start)
        side = dirs[poss]
        x0 = L - BOX_DEPTH if side == 1 else 0
        ax.add_patch(Rectangle((x0, W / 2 - BOX_HALF_WIDTH), BOX_DEPTH, 2 * BOX_HALF_WIDTH,
                               facecolor="yellow", alpha=0.25, edgecolor="yellow", lw=2, zorder=1))
        pcol = HOME_COLOUR if poss == HOME_TAG else AWAY_COLOUR
        ax.add_patch(Rectangle((0, 0), L, W, fill=False, edgecolor=pcol, lw=5, zorder=7))

    # ---- players
    players = _players_in_frame(row)
    for pid, team, x, y, vx, vy in players:
        colour = HOME_COLOUR if team == HOME_TAG else AWAY_COLOUR
        ax.scatter(x, y, s=130, color=colour, edgecolors="black", linewidths=0.6, zorder=3)
        if show_ids:
            ax.text(x, y, str(pid), ha="center", va="center", fontsize=6.5,
                    color="white", weight="bold", zorder=5)
        if vx or vy:
            ax.annotate("", xy=(x + vx * ARROW_SCALE, y + vy * ARROW_SCALE), xytext=(x, y),
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.2), zorder=4)
        if side is not None:
            inside = _box_mask(np.array([[x]]), np.array([[y]]), side)[0, 0]
            if inside:
                ax.scatter(x, y, s=300, facecolors="none", edgecolors="lime", linewidths=2, zorder=4)

    # ---- ball + nearest player
    bx, by = row.get("ball_x", np.nan), row.get("ball_y", np.nan)
    has_ball = pd.notna(bx) and pd.notna(by)
    if has_ball:
        bx_m, by_m = bx * L, by * W
        ax.scatter(bx_m, by_m, s=90, color=BALL_COLOUR, edgecolors="black", linewidths=1.3, zorder=6)
        bvx, bvy = row.get("ball_x_vel", np.nan), row.get("ball_y_vel", np.nan)
        if pd.notna(bvx) and pd.notna(bvy) and (bvx or bvy):
            ax.annotate("", xy=(bx_m + bvx * BALL_ARROW_SCALE, by_m + bvy * BALL_ARROW_SCALE),
                        xytext=(bx_m, by_m), arrowprops=dict(arrowstyle="->", color="yellow", lw=2),
                        zorder=6)
        if players:
            dists = [np.hypot(p[2] - bx_m, p[3] - by_m) for p in players]
            n = players[int(np.argmin(dists))]
            ax.plot([bx_m, n[2]], [by_m, n[3]], color="white", ls="--", lw=1.3, zorder=2)
            ax.scatter(n[2], n[3], s=260, facecolors="none", edgecolors="yellow",
                       linewidths=2, zorder=4)
    att_label = ax.text(1.015, 0.5, "", transform=ax.transAxes, ha="left", va="center",
                        fontsize=12, weight="bold", color="white",
                        bbox=dict(boxstyle="round", facecolor="grey", edgecolor="black"))
    
    
    

    # ---- legend
    def info(t):
        return Line2D([], [], linestyle="none", label=t)

    handles = [
        Line2D([], [], marker="o", ls="none", color=HOME_COLOUR, mec="black", label=f"Home ({HOME_TAG})"),
        Line2D([], [], marker="o", ls="none", color=AWAY_COLOUR, mec="black", label=f"Away ({AWAY_TAG})"),
        Line2D([], [], marker="o", ls="none", color=BALL_COLOUR, mec="black", label="Ball"),
        Line2D([], [], marker="o", ls="none", mfc="none", mec="lime", mew=2, label="Player in attacked box"),
        info(f"Timestamp: {_fmt_timestamp(row.get('timestamp', ''))}"),
    ]
    if c is not None:
        sp = c["set_piece"]
        handles += [
            info(f"Nearest: player {c['nearest_player']} ({c['nearest_team']}, {c['nearest_dist']:.1f} m)"),
            info(f"Possession (10 s rule): {poss}" + ("   <- CHANGE STARTS HERE" if c["possession_change"] else "")),
            info(f"Recomputed in attacked box: att={c['att_in_box']:.0f}, def={c['def_in_box']:.0f}"),
            info(f"Label: {c['label_players_in_box']} players, team {c['label_box_team']}"),
            info(f"Team match: {_flag(c['team_match'])} | Count match: {_flag(c['count_match'])}"),
            info(f"Set piece: {sp if pd.notna(sp) else '-'}"),
        ]
    else:
        handles.append(info("Frame outside the in-play ranges (no sanity info)"))
    
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3,
              frameon=True, fontsize=8.5)
    ax.set_title(f"frame {fid}" + ("" if has_ball else "   -- ball not tracked"))
    return ax



if __name__ == "__main__":
    BOX_DEPTH = 16.5                 # penalty area depth (m)
    BOX_HALF_WIDTH = 20.16 
    df = pd.read_csv("data/dataframe_hackaton_final.csv",low_memory=False) 
    df["time_min"] = pd.to_timedelta(df["timestamp"]).dt.total_seconds() / 60
    MATCH_META = {"offsets": {1: 1702057, 2: 1702058},
              "halves": {1: (1713242, 1792708), 2: (1812866, 1892255)},
              "kickoffs": {1: {"minute": 7, "second": 27, "frame": 11}, 2: {"minute": 75, "second": 49, "frame": 16}}}
    real_halves = {half: (
        (MATCH_META["kickoffs"][half]["minute"] * 60 + MATCH_META["kickoffs"][half]["second"]) * 25 + MATCH_META["kickoffs"][half]["frame"],
        MATCH_META["halves"][half][1] - MATCH_META["offsets"][half])
    for half in [1, 2]}
    chk= sanity_check(df,halves=real_halves)
    dirs = {}
    for half, g in chk.groupby("half"):
        start = g["frame_id"].min()
        w = df[(df["frame_id"] >= start) & (df["frame_id"] < start + 25)]
        h = np.nanmedian(w[[f"{i}_x" for i in HOME_IDS]].to_numpy(float))
        a = np.nanmedian(w[[f"{i}_x" for i in AWAY_IDS]].to_numpy(float))
        dirs[half] = 1 if h < a else -1          # +1 = home attacks towards x = 105
    chk["home_attack_dir"] = chk["half"].map(dirs)
    # plot_minute(df, 12.5)   
    animate_frames(df,30000,40050,chk=chk,step=20,save_path="clip.gif")                             # 12:30
    #anim = animate_minutes(df, 35.0, 36, step=5, save_path="clip.gif")   # 12:00 to 12:15
    plt.show()