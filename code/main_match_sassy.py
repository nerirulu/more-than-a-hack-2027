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

# ==========================================
# QUICK PREVIEW
# ==========================================

START_FRAME = 0

OUTPUT_VIDEO = "sassy_preview.mp4"

CANVAS_W = 1920
CANVAS_H = 1080

PANEL_X = 1260

def put_text_box(
    img,
    text,
    org,
    scale=0.7,
    text_color=(255,255,255),
    bg_color=(40,40,40)
):

    (w, h), _ = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        2
    )

    x, y = org

    cv2.rectangle(
        img,
        (x - 10, y - h - 10),
        (x + w + 10, y + 10),
        bg_color,
        -1
    )

    cv2.putText(
        img,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        text_color,
        2
    )

def overlay_image(
    background,
    overlay,
    x,
    y
):

    h, w = overlay.shape[:2]

    if overlay.shape[2] == 4:

        alpha = overlay[:, :, 3] / 255.0

        for c in range(3):

            background[
                y:y+h,
                x:x+w,
                c
            ] = (
                alpha * overlay[:, :, c]
                +
                (1 - alpha)
                *
                background[
                    y:y+h,
                    x:x+w,
                    c
                ]
            )

    else:

        background[
            y:y+h,
            x:x+w
        ] = overlay
    
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

background_img = cv2.imread(
    "../data/sassy_background.jpg"
)

background_img = cv2.resize(
    background_img,
    (1920, 1080)
)

emoji_img = cv2.imread(
    "../data/emoji.png",
    cv2.IMREAD_UNCHANGED
)

emoji_img = cv2.resize(
    emoji_img,
    (180, 180)
)

barca_img = cv2.imread(
    "../data/viscaelbarca.jfif"
)

barca_img = cv2.resize(
    barca_img,
    (250, 180)
)

print("Background:", background_img.shape)
print("Emoji:", emoji_img.shape)
print("Barca:", barca_img.shape)

# ==========================================
# SASSY ASSETS
# ==========================================

background_img = cv2.imread(
    "../data/sassy_background.jpg"
)

background_img = cv2.resize(
    background_img,
    (1920, 1080)
)

emoji_img = cv2.imread(
    "../data/emoji.png",
    cv2.IMREAD_UNCHANGED
)

emoji_img = cv2.resize(
    emoji_img,
    (180, 180)
)

barca_img = cv2.imread(
    "../data/viscaelbarca.jfif"
)

barca_img = cv2.resize(
    barca_img,
    (250, 180)
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
    desc="💖 VISCA EL BARÇA DEBUGGER 💖",
    unit="frame",
    colour="magenta",
    dynamic_ncols=True
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

    canvas[:] = background_img

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

    put_text_box(
        canvas,
        "MES QUE UN CLUB",
        (PANEL_X, 45),
        scale=1.0,
        bg_color=(255, 0, 200)
    )

    put_text_box(
        canvas,
        f"Frame: {frame_idx}",
        (PANEL_X, 90),
        bg_color=(255,0,200)
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

    put_text_box(
        canvas,
        f"Time: {minutes:02d}:{seconds:02d} ({total_seconds:.1f}s)",
        (PANEL_X, 140),
        bg_color=(255,0,200)
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

    put_text_box(
        canvas,
        (
            f"Ball Speed: {speed:.4f}"
            if speed is not None
            else "Ball Speed: N/A"
        ),
        (PANEL_X, 190),
        bg_color=(255,0,200)
    )

    # =================================================
    # SMART TAGGING
    # =================================================

    put_text_box(
        canvas,
        "SMART TAGGING",
        (PANEL_X, 260),
        bg_color=(255,0,200)
    )

    active_events = get_active_events(
        frame_idx,
        frame_lookup
    )

    y = 310

    if len(active_events) == 0:

        put_text_box(
            canvas,
            "Waiting for football drama...",
            (PANEL_X, y + 80),
            scale=0.6,
            bg_color=(60,60,60)
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

                put_text_box(
                    canvas,
                    tag[:45],
                    (PANEL_X, y),
                    scale=0.55,
                    bg_color=(255,0,200)
                )

            y += 28

    # =================================================
    # SITUATIONS
    # =================================================

    put_text_box(
        canvas,
        "SITUATIONS",
        (PANEL_X, y + 40),
        bg_color=(255,0,200)
    )

    put_text_box(
        canvas,
        "Waiting for football drama...",
        (PANEL_X, y + 80),
        scale=0.6,
        bg_color=(255,0,200)
    )

    # =====================================
    # VISCA EL BARCA
    # =====================================

    put_text_box(
        canvas,
        "VISCA EL BARCA",
        (1320, 835),
        scale=0.9,
        text_color=(255,255,0),
        bg_color=(255,0,200)
    )

    # =====================================
    # FUN IMAGES
    # =====================================

    canvas[
        850:1030,
        1300:1550
    ] = barca_img

    overlay_image(
        canvas,
        emoji_img,
        1650,
        850
    )

    # =====================================
    # PROGRESS BAR
    # =====================================

    progress = (
        frame_idx - START_FRAME
    ) / (
        END_FRAME - START_FRAME
    )

    cv2.rectangle(
        canvas,
        (20, 1070),
        (1900, 1078),
        (80,80,80),
        -1
    )

    cv2.rectangle(
        canvas,
        (20, 1070),
        (
            20 + int(1880 * progress),
            1078
        ),
        (255,0,255),
        -1
    )

    # =====================================
    # PERCENTAGE
    # =====================================

    put_text_box(
        canvas,
        f"{100*progress:.1f}% complete",
        (1600, 820),
        bg_color=(255,0,200)
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