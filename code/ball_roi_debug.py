import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# PARAMETERS
# =====================================================

VIDEO_PATH = "../data/tactical/BAR-ATH_tactical.mp4"
ATD_PATH = "../data/tactical/BAR-ATH_tactical_atd.txt"
HOMO_PATH = "../data/tactical/BAR-ATH_tactical_homography.csv"

# manually choose a frame where ball is still visible
START_FRAME = 12020

# inspect next N frames
N_FRAMES = 20

ROI_SIZE = 30

PEAK_RADIUS = 20

MAX_JUMP_PX = 15

# =====================================================
# LOAD DATA
# =====================================================

homography = pd.read_csv(HOMO_PATH)

with open(ATD_PATH) as f:
    atd_lines = f.readlines()

cap = cv2.VideoCapture(VIDEO_PATH)

# =====================================================
# PROJECT
# =====================================================

def project(H, x, y):

    denom = (
        H[2,0] * x +
        H[2,1] * y +
        H[2,2]
    )

    px = (
        H[0,0] * x +
        H[0,1] * y +
        H[0,2]
    ) / denom

    py = (
        H[1,0] * x +
        H[1,1] * y +
        H[1,2]
    ) / denom

    return int(px), int(py)

def find_ball_disappearances():

    gaps = []

    for frame_idx in range(
        1,
        len(atd_lines)
    ):

        prev_parts = (
            atd_lines[
                frame_idx - 1
            ]
            .strip()
            .split(":")
        )

        curr_parts = (
            atd_lines[
                frame_idx
            ]
            .strip()
            .split(":")
        )

        if (
            len(prev_parts) < 3
            or len(curr_parts) < 3
        ):
            continue

        prev_ball = prev_parts[2]
        curr_ball = curr_parts[2]

        prev_visible = (
            not prev_ball.startswith(
                "NaN"
            )
        )

        curr_visible = (
            curr_ball.startswith(
                "NaN"
            )
        )

        if (
            prev_visible
            and curr_visible
        ):

            gaps.append(
                frame_idx
            )

    return gaps

gaps = find_ball_disappearances()

print(
    "First disappearances:"
)

print(
    gaps[:20]
)

for frame_idx in gaps[:20]:

    print(
        f"{frame_idx-1} visible"
        f" -> "
        f"{frame_idx} missing"
    )

START_FRAME = gaps[19] - 1


# =====================================================
# READ BALL
# =====================================================

def get_ball(frame_number):

    line = atd_lines[frame_number].strip()

    parts = line.split(":")

    if len(parts) < 3:
        return None

    try:

        bx, by, speed = parts[2].split(",")

        if bx == "NaN":
            return None

        return (
            float(bx),
            float(by)
        )

    except:
        return None

# =====================================================
# LAST VISIBLE BALL
# =====================================================

ball0 = get_ball(
    START_FRAME
)

ball1 = get_ball(
    START_FRAME - 1
)

if ball0 is None:
    raise RuntimeError(
        "Ball not visible in START_FRAME"
    )

if ball1 is None:
    raise RuntimeError(
        "Ball not visible in previous frame"
    )

vx = ball0[0] - ball1[0]
vy = ball0[1] - ball1[1]

print("Velocity")
print(vx, vy)

pred_x = ball0[0]
pred_y = ball0[1]

tracked_px = None
tracked_py = None

# =====================================================
# LOOP
# =====================================================

for frame_idx in range(
    START_FRAME,
    START_FRAME + N_FRAMES
):

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_idx
    )

    ret, frame = cap.read()

    if not ret:
        break

    row = homography.iloc[frame_idx]

    H = np.array([
        [row["h00"], row["h01"], row["h02"]],
        [row["h10"], row["h11"], row["h12"]],
        [row["h20"], row["h21"], row["h22"]],
    ])

    ball = get_ball(
        frame_idx
    )

    # -----------------------------------------
    # visible -> use true position
    # -----------------------------------------

    if ball is not None:

        pred_x = ball[0]
        pred_y = ball[1]

        px, py = project(
            H,
            pred_x,
            pred_y
        )

    elif tracked_px is not None:

        px = tracked_px
        py = tracked_py

    else:

        pred_x += vx
        pred_y += vy

        px, py = project(
            H,
            pred_x,
            pred_y
        )

    # -----------------------------------------
    # ROI
    # -----------------------------------------

    x1 = max(
        0,
        px - ROI_SIZE
    )

    x2 = min(
        frame.shape[1],
        px + ROI_SIZE
    )

    y1 = max(
        0,
        py - ROI_SIZE
    )

    y2 = min(
        frame.shape[0],
        py + ROI_SIZE
    )

    roi = frame[
        y1:y2,
        x1:x2
    ]

    if roi.size == 0:
        continue

    roi_rgb = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2RGB
    )

    hsv = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2HSV
    )

    H_channel = hsv[:,:,0]
    S_channel = hsv[:,:,1]
    V_channel = hsv[:,:,2]

    # -----------------------------------------
    # SATURATION INVERSION
    # -----------------------------------------

    ball_response = (
        255 - S_channel
    ).astype(np.uint8)

    # -----------------------------------------
    # GAUSSIAN SMOOTHING
    # -----------------------------------------

    ball_response_blur = cv2.GaussianBlur(
        ball_response,
        (9, 9),
        0
    )

    # -----------------------------------------
    # THRESHOLD
    # -----------------------------------------

    candidate_mask = (
        ball_response_blur >
        np.percentile(
            ball_response_blur,
            98
        )
    )

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            candidate_mask.astype(np.uint8),
            connectivity=8
        )
    )

    centre_x = roi.shape[1] // 2
    centre_y = roi.shape[0] // 2

    best_score = -np.inf

    best_centroid = None

    for i in range(1, num_labels):

        area = stats[
            i,
            cv2.CC_STAT_AREA
        ]

        if area < 3:
            continue

        if area > 150:
            continue

        cx, cy = centroids[i]

        # -----------------------------
        # HARD MOTION CONSTRAINT
        # -----------------------------

        jump_distance = np.sqrt(
            (cx - centre_x)**2 +
            (cy - centre_y)**2
        )

        if jump_distance > MAX_JUMP_PX:
            continue

        mask = (
            labels == i
        )

        blob_response = (
            ball_response_blur.copy()
        )

        blob_response[
            ~mask
        ] = 0

        peak = blob_response.max()

        ys, xs = np.where(mask)

        width = (
            xs.max() - xs.min() + 1
        )

        height = (
            ys.max() - ys.min() + 1
        )

        circularity = (
            min(width, height)
            /
            (max(width, height) + 1e-6)
        )

        score = (
            peak
            * circularity
            / np.sqrt(area)
        )

        if score > best_score:

            best_score = score

            best_centroid = centroids[i]

    VALID_DETECTION = (
        best_centroid is not None
    )

    if VALID_DETECTION:

        max_x = int(
            best_centroid[0]
        )

        max_y = int(
            best_centroid[1]
        )

    # ROI coordinates -> image coordinates

    detected_px = x1 + max_x
    detected_py = y1 + max_y

    print(
        f"Frame {frame_idx} | "
        f"ROI centroid=({max_x},{max_y}) | "
        f"IMG centroid=({detected_px},{detected_py})"
    )

    VALID_DETECTION = (
        best_centroid is not None
    )

    if VALID_DETECTION:

        tracked_x = max_x
        tracked_y = max_y

        tracked_px = detected_px
        tracked_py = detected_py

    else:

        tracked_x = roi.shape[1] // 2
        tracked_y = roi.shape[0] // 2

    # -----------------------------------------
    # GRASS REMOVED
    # -----------------------------------------

    grass_removed = roi_rgb.copy()

    grass_removed[
        S_channel > 70
    ] = 0

    # -----------------------------------------
    # H S V visualizations
    # -----------------------------------------

    H_img = cv2.applyColorMap(
        (
            H_channel * 255 // 180
        ).astype(np.uint8),
        cv2.COLORMAP_HSV
    )

    S_img = cv2.cvtColor(
        S_channel,
        cv2.COLOR_GRAY2RGB
    )

    V_img = cv2.cvtColor(
        V_channel,
        cv2.COLOR_GRAY2RGB
    )

    ball_response_img = cv2.cvtColor(
        ball_response,
        cv2.COLOR_GRAY2RGB
    )

    ball_blur_img = cv2.cvtColor(
        ball_response_blur,
        cv2.COLOR_GRAY2RGB
    )

    candidate_img = roi_rgb.copy()

    # ROI center (prediction)

    cv2.circle(
        candidate_img,
        (
            roi.shape[1] // 2,
            roi.shape[0] // 2
        ),
        5,
        (0,255,0),
        2
    )

    # Detected blob

    cv2.circle(
        candidate_img,
        (
            max_x,
            max_y
        ),
        5,
        (255,0,0),
        2
    )

    if VALID_DETECTION:

        color = (0,255,0)

    else:

        color = (255,0,0)

    cv2.circle(
        candidate_img,
        (
            tracked_x,
            tracked_y
        ),
        5,
        color,
        2
    )

    # -----------------------------------------
    # plotting
    # -----------------------------------------

    fig, ax = plt.subplots(
        2,
        4,
        figsize=(18,8)
    )

    fig.suptitle(
        f"Frame {frame_idx}"
    )

    ax[0,0].imshow(
        roi_rgb
    )
    ax[0,0].set_title(
        "ROI"
    )

    ax[0,1].imshow(
        S_img
    )
    ax[0,1].set_title(
        "Saturation"
    )

    ax[0,2].imshow(
        ball_response_img
    )
    ax[0,2].set_title(
        "255 - Saturation"
    )

    ax[0,3].imshow(
        ball_blur_img
    )
    ax[0,3].set_title(
        "Gaussian Response"
    )

    ax[1,0].imshow(
        grass_removed
    )
    ax[1,0].set_title(
        "Grass Removed"
    )

    ax[1,1].imshow(
        V_img
    )
    ax[1,1].set_title(
        "Value"
    )

    ax[1,2].imshow(
        candidate_mask,
        cmap="gray"
    )
    ax[1,2].set_title(
        "Candidate Mask"
    )

    ax[1,3].imshow(
        candidate_img
    )
    ax[1,3].set_title(
        "Detected Ball"
    )

    for a in ax.ravel():
        a.axis("off")

    plt.tight_layout()
    plt.show()

cap.release()