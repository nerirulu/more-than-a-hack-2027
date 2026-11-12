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


# ==========================================
# CONFIG
# ==========================================

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

START_FRAME = 81000

END_FRAME = 82000

OUTPUT_VIDEO = "debug_match.mp4"


# ==========================================
# LOAD DATA
# ==========================================

homography = pd.read_csv(
    HOMO_PATH
)

with open(ATD_PATH) as f:

    atd_lines = f.readlines()

team_map = load_track_teams(
    TEAM_PATH
)

events = load_smart_tagging(
    SMART_PATH
)

frame_lookup = build_frame_lookup(
    events
)

# ==========================================
# VIDEO
# ==========================================

cap = cv2.VideoCapture(
    VIDEO_PATH
)

fps = cap.get(
    cv2.CAP_PROP_FPS
)

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (1920, 1080)
)

previous_ball = None

# ==========================================
# LOOP
# ==========================================

for frame_idx in tqdm(
    range(START_FRAME, END_FRAME),
    desc="Debug video",
    unit="frame",
    colour="green"
):

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_idx
    )

    ret, frame = cap.read()

    if not ret:
        break

    # ======================================
    # HOMOGRAPHY
    # ======================================

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

    # ======================================
    # LOAD PLAYERS
    # ======================================

    players, ball = (
        get_players_and_ball(
            atd_lines,
            frame_idx,
            team_map
        )
    )

    # ======================================
    # VIDEO OVERLAY
    # ======================================

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

    overlay = draw_tracking_state(
        frame,
        H,
        overlay_players,
        ball_position=ball
    )

    # ======================================
    # TACTICAL BOARD
    # ======================================

    pitch_img = render_pitch_cv(
        overlay_players,
        ball
    )

    pitch_img = cv2.resize(
        pitch_img,
        (700, 350)
    )

    # ======================================
    # CANVAS
    # ======================================

    canvas = np.zeros(
        (
            1080,
            1920,
            3
        ),
        dtype=np.uint8
    )

    canvas[:] = 30

    video_view = cv2.resize(
        overlay,
        (1200, 675)
    )

    canvas[
        20:695,
        20:1220
    ] = video_view

    canvas[
        710:1060,
        20:720
    ] = pitch_img

    # ======================================
    # INFO PANEL
    # ======================================

    panel_x = 1260

    cv2.putText(
        canvas,
        f"Frame: {frame_idx}",
        (panel_x, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,255),
        2
    )

    seconds = frame_idx / 25

    cv2.putText(
        canvas,
        f"Time: {seconds:.1f}s",
        (panel_x, 100),
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
            (panel_x, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,255),
            2
        )

    # ======================================
    # SMART TAGGING
    # ======================================

    cv2.putText(
        canvas,
        "SMART TAGGING",
        (panel_x, 220),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0,255,255),
        2
    )

    active_events = (
        get_active_events(
            frame_idx,
            frame_lookup
        )
    )

    y = 260

    for event in active_events[:12]:

        label = str(
            event.get(
                "code",
                ""
            )
        )

        if (
            event.get("Team")
            is not None
        ):

            label += (
                f" | "
                f"{event['Team']}"
            )

        cv2.putText(
            canvas,
            label[:40],
            (panel_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255,255,255),
            1
        )

        y += 30

    # ======================================
    # PLACEHOLDER FOR
    # THROW-IN/CORNER/GOALKICK
    # ======================================

    cv2.putText(
        canvas,
        "SITUATIONS",
        (panel_x, y + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0,255,255),
        2
    )

    cv2.putText(
        canvas,
        "None",
        (panel_x, y + 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    writer.write(
        canvas
    )

# ==========================================
# FINISH
# ==========================================

cap.release()

writer.release()

print()
print("Saved:")
print(OUTPUT_VIDEO)