import cv2
import numpy as np
import pandas as pd

from utils.atd_utils import get_players_and_ball
from utils.atd_team_utils import load_track_teams

from utils.smart_tagging_utils import (
    load_smart_tagging,
    build_frame_lookup,
    get_active_events
)

from utils.pitch_projection_utils import (
    draw_tracking_state
)

from utils.football_pitch import (
    render_pitch_cv
)

from utils.ball_utils import (
    ball_speed
)

from tqdm import tqdm

# =====================================================
# CONFIG
# =====================================================

VIDEO_PATH = "../data/tactical/BAR-ATH_tactical.mp4"

ATD_PATH = "../data/tactical/BAR-ATH_tactical_atd.txt"

HOMO_PATH = "../data/tactical/BAR-ATH_tactical_homography.csv"

SMART_PATH = (
    "../data/tactical/"
    "BAR-ATH_tactical_smart_tagging.xml"
)

TEAM_PATH = (
    "../data/broadcast/"
    "BAR-ATH_broadcast_atd.xml"
)

START_FRAME = 0

OUTPUT_VIDEO = "full_debug_match.mp4"

CANVAS_W = 1920
CANVAS_H = 1080

PANEL_X = 1260

# =====================================================
# LOAD DATA
# =====================================================

print("Loading homography...")
homography = pd.read_csv(HOMO_PATH)

print("Loading ATD...")
with open(ATD_PATH) as f:
    atd_lines = f.readlines()

END_FRAME = len(atd_lines)

print("Loading teams...")
team_map = load_track_teams(
    TEAM_PATH
)

print("Loading smart tagging...")
events = load_smart_tagging(
    SMART_PATH
)

frame_lookup = build_frame_lookup(
    events
)

# =====================================================
# VIDEO
# =====================================================

cap = cv2.VideoCapture(
    VIDEO_PATH
)

fps = cap.get(
    cv2.CAP_PROP_FPS
)

cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    START_FRAME
)

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (
        CANVAS_W,
        CANVAS_H
    )
)

# =====================================================
# CANVAS REUSE
# =====================================================

canvas = np.zeros(
    (
        CANVAS_H,
        CANVAS_W,
        3
    ),
    dtype=np.uint8
)

previous_ball = None

# =====================================================
# MAIN LOOP
# =====================================================

for frame_idx in tqdm(
    range(START_FRAME, END_FRAME),
    desc="Rendering video",
    unit="frame"
):

    ret, frame = cap.read()

    if not ret:
        break

    # =================================================
    # HOMOGRAPHY
    # =================================================

    row = homography.iloc[
        frame_idx
    ]

    H = np.array(
        [
            [
                row["h00"],
                row["h01"],
                row["h02"]
            ],
            [
                row["h10"],
                row["h11"],
                row["h12"]
            ],
            [
                row["h20"],
                row["h21"],
                row["h22"]
            ]
        ]
    )

    # =================================================
    # PLAYERS + BALL
    # =================================================

    players, ball = get_players_and_ball(
        atd_lines,
        frame_idx,
        team_map
    )

    # =================================================
    # PLAYER FORMAT
    # =================================================

    overlay_players = []

    for player in players:

        overlay_players.append(
            {
                "id": player["id"],
                "team": player["team"],
                "x": player["x"],
                "y": player["y"]
            }
        )

    # =================================================
    # VIDEO OVERLAY
    # =================================================

    overlay = draw_tracking_state(
        frame,
        H,
        overlay_players,
        ball_position=ball
    )

    # =================================================
    # PITCH
    # =================================================

    pitch_img = render_pitch_cv(
        overlay_players,
        ball
    )

    pitch_img = cv2.resize(
        pitch_img,
        (
            700,
            350
        )
    )

    # =================================================
    # RESET CANVAS
    # =================================================

    canvas[:] = 25

    # =================================================
    # VIDEO PANEL
    # =================================================

    video_view = cv2.resize(
        overlay,
        (
            1200,
            675
        )
    )

    canvas[
        20:695,
        20:1220
    ] = video_view

    # =================================================
    # PITCH PANEL
    # =================================================

    canvas[
        710:1060,
        20:720
    ] = pitch_img

    # =================================================
    # GENERAL INFO
    # =================================================

    cv2.putText(
        canvas,
        "MATCH DEBUG",
        (PANEL_X, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0,255,255),
        2
    )

    cv2.putText(
        canvas,
        f"Frame: {frame_idx}",
        (PANEL_X, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,255),
        2
    )

    total_seconds = (
        frame_idx / fps
    )

    minutes = int(
        total_seconds // 60
    )

    seconds = int(
        total_seconds % 60
    )

    cv2.putText(
        canvas,
        f"Time: {minutes:02d}:{seconds:02d}",
        (PANEL_X, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,255),
        2
    )

    speed = ball_speed(
        ball,
        previous_ball
    )

    previous_ball = (
        ball.copy()
        if ball is not None
        else None
    )

    if speed is not None:

        cv2.putText(
            canvas,
            f"Ball Speed: {speed:.4f}",
            (PANEL_X, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,255),
            2
        )

    else:

        cv2.putText(
            canvas,
            "Ball Speed: N/A",
            (PANEL_X, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,255),
            2
        )

    # =================================================
    # SMART TAGGING
    # =================================================

    cv2.putText(
        canvas,
        "SMART TAGGING",
        (PANEL_X, 260),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0,255,255),
        2
    )

    active_events = get_active_events(
        frame_idx,
        frame_lookup
    )

    y = 310

    if len(active_events) == 0:

        cv2.putText(
            canvas,
            "None",
            (PANEL_X, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,255),
            1
        )

        y += 30

    else:

        for event in active_events[:12]:

            code = str(
                event.get(
                    "code",
                    ""
                )
            )

            team = str(
                event.get(
                    "Team",
                    ""
                )
            )

            tag = code

            if team not in [
                "",
                "nan",
                "None"
            ]:

                tag += (
                    f" | {team}"
                )

            cv2.putText(
                canvas,
                tag[:45],
                (PANEL_X, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255,255,255),
                1
            )

            y += 28

    # =================================================
    # SITUATIONS
    # =================================================

    cv2.putText(
        canvas,
        "SITUATIONS",
        (PANEL_X, y + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0,255,255),
        2
    )

    cv2.putText(
        canvas,
        "None",
        (PANEL_X, y + 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    # =================================================
    # WRITE
    # =================================================

    writer.write(
        canvas
    )

# =====================================================
# FINISH
# =====================================================

cap.release()

writer.release()

print()
print("Finished.")
print("Saved:")
print(OUTPUT_VIDEO)