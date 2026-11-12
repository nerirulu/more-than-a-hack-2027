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

START_FRAME = 12020
N_FRAMES = 20

ROI_SIZE = 30

# controls how strongly distance is penalized
DISTANCE_SIGMA = 12

MAX_TRACK_JUMP = 10

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
        H[2, 0] * x
        + H[2, 1] * y
        + H[2, 2]
    )

    px = (
        H[0, 0] * x
        + H[0, 1] * y
        + H[0, 2]
    ) / denom

    py = (
        H[1, 0] * x
        + H[1, 1] * y
        + H[1, 2]
    ) / denom

    return int(px), int(py)

# =====================================================
# FIND DISAPPEARANCES
# =====================================================

def find_ball_disappearances():

    gaps = []

    for frame_idx in range(1, len(atd_lines)):

        prev_parts = (
            atd_lines[frame_idx - 1]
            .strip()
            .split(":")
        )

        curr_parts = (
            atd_lines[frame_idx]
            .strip()
            .split(":")
        )

        if len(prev_parts) < 3 or len(curr_parts) < 3:
            continue

        prev_visible = not prev_parts[2].startswith("NaN")
        curr_missing = curr_parts[2].startswith("NaN")

        if prev_visible and curr_missing:
            gaps.append(frame_idx)

    return gaps

gaps = find_ball_disappearances()

print("First disappearances:")
print(gaps[:20])

for g in gaps[:20]:
    print(f"{g-1} visible -> {g} missing")

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
# INITIAL BALL
# =====================================================

ball0 = get_ball(START_FRAME)
ball1 = get_ball(START_FRAME - 1)

if ball0 is None:
    raise RuntimeError("Ball not visible in START_FRAME")

if ball1 is None:
    raise RuntimeError("Ball not visible in previous frame")

vx = ball0[0] - ball1[0]
vy = ball0[1] - ball1[1]

print("Velocity:", vx, vy)

pred_x = ball0[0]
pred_y = ball0[1]

tracked_px = None
tracked_py = None

# =====================================================
# MAIN LOOP
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

    ball = get_ball(frame_idx)

    # -------------------------------------------------
    # prediction
    # -------------------------------------------------

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

    # -------------------------------------------------
    # ROI
    # -------------------------------------------------

    x1 = max(0, px - ROI_SIZE)
    x2 = min(frame.shape[1], px + ROI_SIZE)

    y1 = max(0, py - ROI_SIZE)
    y2 = min(frame.shape[0], py + ROI_SIZE)

    roi = frame[y1:y2, x1:x2]

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

    H_channel = hsv[:, :, 0]
    S_channel = hsv[:, :, 1]
    V_channel = hsv[:, :, 2]

    # -------------------------------------------------
    # RESPONSE
    # -------------------------------------------------

    ball_response = (
        255 - S_channel
    ).astype(np.uint8)

    ball_response_blur = cv2.GaussianBlur(
        ball_response,
        (9, 9),
        0
    )

    threshold = np.percentile(
        ball_response_blur,
        98
    )

    candidate_mask = (
        ball_response_blur > threshold
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
    best_mask = None

    # -------------------------------------------------
    # SCORING
    # -------------------------------------------------

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

        distance = np.sqrt(
            (cx - centre_x) ** 2 +
            (cy - centre_y) ** 2
        )

        mask = labels == i

        blob_response = (
            ball_response_blur.copy()
        )

        blob_response[
            ~mask
        ] = 0

        peak = float(
            blob_response.max()
        )

        ys, xs = np.where(mask)

        if len(xs) == 0:
            continue

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

        distance_weight = np.exp(
            -(distance ** 2)
            /
            (2 * DISTANCE_SIGMA ** 2)
        )

        score = (
            peak
            * (circularity ** 2)
            * distance_weight
            /
            np.sqrt(area)
        )

        print(
            f"blob={i}"
            f" area={area}"
            f" peak={peak:.1f}"
            f" circ={circularity:.2f}"
            f" dist={distance:.1f}"
            f" dw={distance_weight:.2f}"
            f" score={score:.2f}"
        )

        if score > best_score:

            best_score = score
            best_centroid = centroids[i]
            best_mask = mask

    # -------------------------------------------------
    # DETECTION
    # -------------------------------------------------

    VALID_DETECTION = (
        best_centroid is not None
    )

    if VALID_DETECTION:

        max_x = int(best_centroid[0])
        max_y = int(best_centroid[1])

        detected_px = x1 + max_x
        detected_py = y1 + max_y

        tracked_px = detected_px
        tracked_py = detected_py

        tracked_x = max_x
        tracked_y = max_y

        print(
            f"WINNER score={best_score:.2f}"
        )

    else:

        tracked_x = centre_x
        tracked_y = centre_y

        print("NO VALID DETECTION")

    # -------------------------------------------------
    # VISUALIZATION
    # -------------------------------------------------

    candidate_img = roi_rgb.copy()

    if best_mask is not None:

        overlay = candidate_img.copy()

        overlay[best_mask] = [255, 0, 0]

        candidate_img = cv2.addWeighted(
            candidate_img,
            0.6,
            overlay,
            0.4,
            0
        )

    cv2.circle(
        candidate_img,
        (centre_x, centre_y),
        4,
        (0, 255, 0),
        2
    )

    if VALID_DETECTION:

        cv2.circle(
            candidate_img,
            (max_x, max_y),
            6,
            (255, 0, 0),
            2
        )

    fig, ax = plt.subplots(
        2,
        3,
        figsize=(16, 8)
    )

    fig.suptitle(
        f"Frame {frame_idx}"
    )

    ax[0,0].imshow(roi_rgb)
    ax[0,0].set_title("ROI")

    ax[0,1].imshow(
        S_channel,
        cmap="gray"
    )
    ax[0,1].set_title("Saturation")

    ax[0,2].imshow(
        ball_response,
        cmap="gray"
    )
    ax[0,2].set_title("255-S")

    ax[1,0].imshow(
        ball_response_blur,
        cmap="gray"
    )
    ax[1,0].set_title("Gaussian")

    ax[1,1].imshow(
        candidate_mask,
        cmap="gray"
    )
    ax[1,1].set_title("Candidate Mask")

    ax[1,2].imshow(
        candidate_img
    )
    ax[1,2].set_title("Winning Blob")

    for a in ax.ravel():
        a.axis("off")

    plt.tight_layout()
    plt.show()

cap.release()