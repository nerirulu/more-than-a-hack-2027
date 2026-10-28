# -*- coding: utf-8 -*-
"""Visualise the tracked players and ball for a window of the match.

Reads the tactical ATD raw text file (same format the baseline uses) and renders
every Nth frame of a chosen window as a pitch overlay, so you can *see* what the
tracking is doing: where players cluster, where the ball disappears, and
optionally look ahead into the future to draw faint trails.

DATA SOURCES
------------
All data comes from Metrica Sports' computer-vision pipeline, which processes
two camera feeds of the same LaLiga match (Barcelona vs Athletic Club,
2024-25, matchday 2).

Tactical video      -> data/tactical/BAR-ATH_tactical.mp4
                      The frame-index authority: submission row i == video frame i.
Broadcast video     -> data/broadcast/BAR-ATH_broadcast.mp4
                      Optional extra input. Own timeline (164 488 frames).
ATD (tactical)      -> data/tactical/BAR-ATH_tactical_atd.xml + .txt
                      Per-frame (x, y, speed) per track slot.
                      Coordinates are normalised 0-1, not metres.
Homography          -> data/tactical/BAR-ATH_tactical_homography.csv
                      Per-frame 3x3 camera-to-pitch matrix (0-1 -> pixels).
Smart Tagging       -> data/tactical/BAR-ATH_tactical_smart_tagging.xml
                      638 tactical phases with team tags.

FRAME ALIGNMENT (tactical)
--------------------------
  submission row i      -> video frame i
  ATD record n          -> video frame n - 1  (ATD counts from 1)
  Smart Tagging         -> seconds * 25
  homography            -> already a submission row, frame = 0..199318

GRADED FRAMES
-------------
  1st half:  video frames 11185..90651
  2nd half:  video frames 110808..190197
  All others are ungraded.

HOW TO USE
----------
  python starter/visualize_tracking.py               # first graded minute, 1 frame per 5 s
  python starter/visualize_tracking.py --frames 11185 12684 --step 25   # every second
  python starter/visualize_tracking.py --minutes 1.5 --step 25          # every second
  python starter/visualize_tracking.py --frames 11185 12684 --step 1     # every frame

The output PNGs are a visual aid -- not a submission, not scored.
"""

import argparse
import os
import subprocess
import sys
import glob
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib
matplotlib.use(os.environ.get("MPL_BACKEND", "Agg"))
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require  # noqa: E402

FPS = GAMES[GAME]["fps"]
PITCH_LENGTH = GAMES[GAME]["pitch"][0]
PITCH_WIDTH = GAMES[GAME]["pitch"][1]
IMAGE_DIR = os.path.join(ROOT, "starter", "visual_frames")
FIG_SIZE = (12, 8)
SAVE_DPI = 110


def get_atd_paths(game=GAME, data_dir=None):
    """Return (meta_path, raw_path) for the tactical ATD, or raise a readable error."""
    require(["atd_meta", "atd_raw"], game, data_dir, "tactical")  # noqa: F821
    from games import DATA_DIR
    meta_path = os.path.join(DATA_DIR, "tactical", "BAR-ATH_tactical_atd.xml")
    raw_path = os.path.join(DATA_DIR, "tactical", "BAR-ATH_tactical_atd.txt")
    for name, p in (("meta", meta_path), ("raw", raw_path)):
        if not os.path.isfile(p):
            raise IOError(
                "ATD file not found: {}\n\n"
                "The match data is gated -- see DATA.md. Unpack the bundle into "
                "./data or point MTAH_DATA_DIR at where you put it.".format(p)
            )
    return meta_path, raw_path


def load_team_map(meta_path):
    """Parse the ATD metadata XML and return {track_slot: team_id}.

    Team ids are 0 (home) / 1 (away), matching the order teams appear in
    the XML: first team = home, second team = away.
    """
    with open(meta_path) as handle:
        tree = ET.parse(handle)

    root = tree.getroot()
    teams_elem = root.find("Metadata/Teams")
    team_id_to_side = {}
    if teams_elem is not None:
        for idx, team in enumerate(teams_elem.findall("Team")):
            team_id_to_side[team.get("id")] = idx  # 0 = home, 1 = away

    players_elem = root.find("Metadata/Players")
    track_to_side = {}
    if players_elem is not None:
        for player in players_elem.findall("Player"):
            slot = int(player.get("id"))
            team_id = player.get("teamId")
            track_to_side[slot] = team_id_to_side.get(team_id, NOBODY)

    return track_to_side


def read_atd_window(frame_range, game=GAME, data_dir=None):
    """Stream the tactical ATD and return per-frame arrays for the requested
    video frames (0-based range, inclusive).

    Returns a dict with:
      frame_ids  -- list of 1-based frame ids in the ATD file
      players    -- list of (n_tracks, 2) arrays of (x, y) in 0-1, None where missing
      ball       -- list of (2,) arrays of ball (x, y), None where missing
      n_tracks   -- number of track slots in this export
      team_map   -- dict {track_slot: 0/1/-1} home/away/nobody
    """
    meta_path, raw_path = get_atd_paths(game, data_dir)
    team_map = load_team_map(meta_path)

    with open(meta_path) as handle:
        tree = ET.parse(handle)
    n_tracks = len(list(tree.getroot().iter("Player")))

    frame_ids = []
    players = []
    ball = []
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            frame_id = int(parts[0])
            if frame_id < frame_range[0] or frame_id > frame_range[1]:
                continue

            fields = parts[1].split(";")
            if len(fields) != n_tracks:
                raise ValueError(
                    "Expected {} track slots, got {}. Line: {}".format(
                        n_tracks, len(fields), line[:80]
                    )
                )

            frame_ids.append(frame_id)
            players.append([None] * n_tracks)
            for slot, group in enumerate(fields):
                if group[0] == MISSING:
                    continue
                comma = group.index(",")
                x = float(group[:comma])
                y = float(group[comma + 1:group.index(",", comma + 1)])
                players[-1][slot] = (x, y)

            # Ball is parts[2] -- format "x,y,speed" or "NaN,NaN,NaN"
            ball_field = parts[2]
            if ball_field[0] == MISSING:
                ball.append(None)
            else:
                comma = ball_field.index(",")
                bx = float(ball_field[:comma])
                by = float(ball_field[comma + 1:ball_field.index(",", comma + 1)])
                ball.append((bx, by))

    return {
        "frame_ids": frame_ids,
        "players": players,
        "ball": ball,
        "n_tracks": n_tracks,
        "team_map": team_map,
    }
# Team colours
HOME_COLOUR = "#d62728"   # Barcelona (home)
AWAY_COLOUR = "#1f77b4"   # Atletic Club (away)
BALL_COLOUR = "#ffffff"
PLAYER_SIZE = 70
BALL_SIZE = 80

HOME, AWAY = 0, 1
NOBODY = -1
LABELS = np.array(["H", "A", "D"])
FIELD_SEP = ":"
GROUP_SEP = ";"
MISSING = "N"  # first char of "NaN", all we need to test for
STEP_DEFAULT = 25  # one frame per second at 25 fps (was 125 = 5 seconds, too sparse)

def draw_frame(ax, frame_pos, vis):
    """Draw one frame onto ax. frame_pos is the index into vis's lists."""
    frame_idx = vis["frame_ids"][frame_pos]
    ax.clear()
    ax.set_xlim(-3, PITCH_LENGTH + 3)
    ax.set_ylim(-3, PITCH_WIDTH + 3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        "video frame {:>6}  (ATD record {})".format(frame_idx, frame_idx),
        fontsize=10,
    )

    # Pitch background
    from matplotlib.patches import Rectangle, Circle
    ax.add_patch(Rectangle(
        (0, 0), PITCH_LENGTH, PITCH_WIDTH,
        facecolor="#3f8f4f", edgecolor="#ffffff", zorder=0,
    ))
    ax.plot([PITCH_LENGTH / 2, PITCH_LENGTH / 2], [0, PITCH_WIDTH],
            color="#ffffff", zorder=1)
    ax.add_patch(Circle(
        (PITCH_LENGTH / 2, PITCH_WIDTH / 2), 9.15,
        fill=False, edgecolor="#ffffff", zorder=1,
    ))
    for side in (0, 1):
        for depth, box_width in ((16.5, 40.3), (5.5, 18.3)):
            x = 0 if side == 0 else PITCH_LENGTH - depth
            ax.add_patch(Rectangle(
                (x, (PITCH_WIDTH - box_width) / 2), depth, box_width,
                fill=False, edgecolor="#ffffff", zorder=1,
            ))

    # Players -- coloured by team
    player_positions = vis["players"][frame_pos]
    team_map = vis["team_map"]
    for slot in range(vis["n_tracks"]):
        slot_data = player_positions[slot]
        if slot_data is None:
            continue
        px, py = slot_data
        side = team_map.get(slot, NOBODY)
        colour = HOME_COLOUR if side == HOME else AWAY_COLOUR if side == AWAY else "#808080"
        ax.scatter(px * PITCH_LENGTH, py * PITCH_WIDTH, s=PLAYER_SIZE, zorder=3,
                   facecolor=colour, edgecolor="black", linewidth=0.5)

    # Ball -- white dot
    ball_pos = vis["ball"][frame_pos]
    if ball_pos is not None:
        bx, by = ball_pos
        ax.scatter(bx * PITCH_LENGTH, by * PITCH_WIDTH, s=BALL_SIZE, zorder=4,
                   facecolor=BALL_COLOUR, edgecolor="black", linewidth=1.2)

    ax.set_title(
        ax.get_title() + ("   -- ball tracked" if ball_pos is not None else "   -- ball NOT tracked")
    )


def render_window(vis, frame_indices, output_dir, trail_frames=30):
    """Render every frame in frame_indices as a PNG pitch overlay.

    trail_frames: when set to N > 0, look ahead N future frames to draw a
    faint trail of where players and the ball have been -- this is the
    "future data" trick you mentioned, purely for display.
    """
    import matplotlib
    matplotlib.use(os.environ.get("MPL_BACKEND", "Agg"))
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    n = len(frame_indices)
    print("Rendering {} frames to {}".format(n, output_dir))

    for k, frame_idx in enumerate(frame_indices):
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        draw_frame(ax, k, vis)
        # Draw a faint trail on the same axes (looks ahead into the future)
        if trail_frames > 0 and k + trail_frames < n:
            start = max(0, k)
            end = min(n, k + trail_frames + 1)
            for j in range(start, end):
                player_positions = vis["players"][j]
                for slot in range(vis["n_tracks"]):
                    slot_data = player_positions[slot]
                    if slot_data is None:
                        continue
                    x, y = slot_data
                    ax.plot(x * PITCH_LENGTH, y * PITCH_WIDTH,
                            color="gray", alpha=0.15, linewidth=0.8, zorder=2)
            ball_trail = vis["ball"][k + 10]  # mid-point of trail window
            if ball_trail is not None:
                bx, by = ball_trail
                ax.plot(bx * PITCH_LENGTH, by * PITCH_WIDTH,
                        color="gray", alpha=0.15, linewidth=0.8, zorder=2)

        out_path = os.path.join(
            output_dir, "frame_{:05d}.png".format(frame_idx)
        )
        fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        if (k + 1) % 50 == 0:
            print("  wrote {} ({:.0f}% of {}, frame {})".format(
                os.path.basename(out_path), 100 * (k + 1) / n,
                n, frame_idx,
            ))

    # Assemble a GIF animation from the PNGs (requires ffmpeg or PIL)
    png_files = sorted(glob.glob(os.path.join(output_dir, "frame_*.png")))
    if png_files:
        generate_animation(png_files, output_dir, fps=max(1, FPS // 5))

    return output_dir

def generate_animation(png_files, output_dir, fps=5):
    """Create an MP4 (ffmpeg) or GIF from the PNG sequence."""
    # Try ffmpeg first (MP4)
    mp4_path = os.path.join(output_dir, "animation.mp4")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-framerate", str(fps),
             "-i", os.path.join(output_dir, "frame_%05d.png"),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", mp4_path],
            check=True, capture_output=True, timeout=120,
        )
        print("  wrote {}".format(mp4_path))
        return
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    # Fall back to matplotlib GIF that actually loads the rendered PNGs
    try:
        import matplotlib.animation as animation
        from PIL import Image

        frames = [np.array(Image.open(p)) for p in png_files]

        fig, ax = plt.subplots(figsize=FIG_SIZE)
        im = ax.imshow(frames[0], aspect="equal")
        ax.axis("off")

        def update(k):
            im.set_array(frames[k])
            ax.set_title(os.path.basename(png_files[k]).replace(".png", ""))
            return [im]

        anim = animation.FuncAnimation(
            fig, update, frames=len(frames), interval=1000 // fps, blit=False
        )
        gif_path = os.path.join(output_dir, "animation.gif")
        anim.save(gif_path, writer="pillow", fps=fps)
        plt.close(fig)
        print("  wrote {}".format(gif_path))
    except Exception as exc:  # pragma: no cover
        print("  animation fallback failed: {}".format(exc))


def main():
    parser = argparse.ArgumentParser(
        description="Visualise tracking for a window of the match"
    )
    parser.add_argument(
        "--frames", type=int, nargs=2, metavar=("START", "END"),
        help="video frames (inclusive) to render",
    )
    parser.add_argument(
        "--minutes", type=float,
        help="window of this many minutes, centred on frame 90000",
    )
    parser.add_argument(
        "--step", type=int, default=STEP_DEFAULT,
        help="render every Nth frame (default: {} = 1 second at 25 fps)".format(
            STEP_DEFAULT),
    )
    parser.add_argument(
        "--output", default="visual_frames",
        help="output directory for PNGs",
    )
    parser.add_argument(
        "--trail-frames", type=int, default=30,
        help="how many future frames to look ahead for trails",
    )
    args = parser.parse_args()

    if args.frames:
        frame_range = (args.frames[0], args.frames[1])
    elif args.minutes:
        centre = 90000
        n_frames = int(args.minutes * FPS)
        frame_range = (centre - n_frames // 2, centre + n_frames // 2)
    else:
        frame_range = (11185, 11185 + FPS * 60)  # default: first graded minute

    # Build the list of frame indices to render (stepped)
    all_frames = list(range(frame_range[0], frame_range[1] + 1))
    frame_indices = all_frames[::args.step]

    print("Rendering {} frames ({}..{}, step={})".format(
        len(frame_indices), frame_range[0], frame_range[1], args.step))

    try:
        vis = read_atd_window(frame_range)
        print("Loaded {} valid frames".format(len(vis["frame_ids"])))
    except Exception as e:
        print("ERROR loading ATD: {}".format(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)

    try:
        render_window(
            vis, frame_indices, args.output, trail_frames=args.trail_frames,
        )
        print("Done. PNGs in {}".format(args.output))
    except Exception as e:
        print("ERROR rendering: {}".format(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()