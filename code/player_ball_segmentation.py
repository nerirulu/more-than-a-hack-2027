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

FRAME = 10700

# choose which player to inspect
PLAYER_IDX = 10

# =====================================================
# LOAD FRAME
# =====================================================

cap = cv2.VideoCapture(VIDEO_PATH)

cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME)

ret, frame = cap.read()

if not ret:
    raise RuntimeError("Could not read frame")

cap.release()

frame_rgb = cv2.cvtColor(
    frame,
    cv2.COLOR_BGR2RGB
)

# =====================================================
# LOAD HOMOGRAPHY
# =====================================================

homography = pd.read_csv(HOMO_PATH)

row = homography.iloc[FRAME]

H = np.array([
    [row["h00"], row["h01"], row["h02"]],
    [row["h10"], row["h11"], row["h12"]],
    [row["h20"], row["h21"], row["h22"]],
])

# =====================================================
# LOAD ATD FRAME
# =====================================================

target_frame_id = FRAME + 1

with open(ATD_PATH) as f:

    for line in f:

        if line.startswith(f"{target_frame_id}:"):

            atd_line = line.strip()
            break

players = atd_line.split(":")[1].split(";")

# =====================================================
# PROJECT PLAYERS
# =====================================================

projected_points = []

for player in players:

    if player.startswith("NaN"):
        continue

    try:

        x, y, speed = player.split(",")

        x = float(x)
        y = float(y)

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

        img_h, img_w = frame.shape[:2]

        if (
            px >= 0 and
            px < img_w and
            py >= 0 and
            py < img_h
        ):
            projected_points.append(
                (int(px), int(py))
            )

    except:
        pass

print("Tracked players:", len(projected_points))

# =====================================================
# FULL FRAME SEGMENTATION MAP
# =====================================================

segmentation = np.zeros(
    (*frame.shape[:2], 3),
    dtype=np.uint8
)

colors = plt.cm.tab20(
    np.linspace(0, 1, len(projected_points))
)

for idx, (px, py) in enumerate(projected_points):

    height = int(
        np.interp(
            py,
            [250, frame.shape[0]],
            [20, 60]
        )
    )

    width = int(height * 0.6)

    x1 = max(0, px - width // 2)
    x2 = min(frame.shape[1], px + width // 2)

    y1 = max(0, py - height)
    y2 = min(frame.shape[0], py + 5)

    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        continue

    roi_h, roi_w = roi.shape[:2]

    # ==========================================
    # KMEANS
    # ==========================================

    pixels = roi.reshape((-1,3))
    pixels = np.float32(pixels)

    K = 4

    criteria = (
        cv2.TERM_CRITERIA_EPS +
        cv2.TERM_CRITERIA_MAX_ITER,
        50,
        0.2
    )

    _, labels_k, centers = cv2.kmeans(
        pixels,
        K,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )

    cluster_map = labels_k.reshape(
        roi_h,
        roi_w
    )

    # find grass cluster

    scores = []

    for center in centers:

        b, g, r = center

        score = (
            g
            - 0.5 * r
            - 0.5 * b
        )

        scores.append(score)

    grass_cluster = np.argmax(scores)

    player_mask = np.uint8(
        cluster_map != grass_cluster
    )

    # ==========================================
    # CLEANUP
    # ==========================================

    kernel = np.ones((3,3), np.uint8)

    player_mask = cv2.morphologyEx(
        player_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # ==========================================
    # COLORIZE THIS PLAYER
    # ==========================================

    color = (
        np.array(
            colors[idx][:3]
        ) * 255
    ).astype(np.uint8)

    roi_seg = segmentation[
        y1:y2,
        x1:x2
    ]

    roi_seg[
        player_mask > 0
    ] = color

# =====================================================
# OVERLAY
# =====================================================

overlay = cv2.addWeighted(
    frame_rgb,
    0.6,
    segmentation,
    0.9,
    0
)

# =====================================================
# DISPLAY
# =====================================================

fig, ax = plt.subplots(
    1,
    3,
    figsize=(20,8)
)

ax[0].imshow(frame_rgb)
ax[0].set_title("Original Frame")

ax[1].imshow(segmentation)
ax[1].set_title("Player Segmentation Map")

ax[2].imshow(overlay)
ax[2].set_title("Overlay")

for a in ax:
    a.axis("off")

plt.tight_layout()
plt.show()


def extract_player_hsv(frame, px, py):

    global segmentation_overlay

    player_height = int(
        np.interp(
            py,
            [250, frame.shape[0]],
            [30, 80]
        )
    )

    player_width = int(
        player_height * 0.6
    )

    x1 = max(
        0,
        px - player_width // 2
    )

    x2 = min(
        frame.shape[1],
        px + player_width // 2
    )

    y1 = max(
        0,
        py - int(player_height * 1.8)
    )

    y2 = min(
        frame.shape[0],
        py + 5
    )

    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return None

    roi_h, roi_w = roi.shape[:2]

    pixels = roi.reshape(
        (-1, 3)
    ).astype(np.float32)

    K = 4

    criteria = (
        cv2.TERM_CRITERIA_EPS +
        cv2.TERM_CRITERIA_MAX_ITER,
        50,
        0.2
    )

    try:

        _, labels_k, centers = cv2.kmeans(
            pixels,
            K,
            None,
            criteria,
            10,
            cv2.KMEANS_RANDOM_CENTERS
        )

    except:
        return None

    cluster_map = labels_k.reshape(
        roi_h,
        roi_w
    )

    # -----------------------------------
    # Find grass cluster
    # -----------------------------------

    scores = []

    for center in centers:

        b, g, r = center

        score = (
            g
            - 0.5 * r
            - 0.5 * b
        )

        scores.append(score)

    grass_cluster = np.argmax(
        scores
    )

    # -----------------------------------
    # Build player mask
    # -----------------------------------

    foot_x = roi_w // 2
    foot_y = roi_h - 5

    player_mask = np.zeros(
        (roi_h, roi_w),
        dtype=np.uint8
    )

    for c in range(K):

        if c == grass_cluster:
            continue

        cluster = (
            cluster_map == c
        )

        area = np.sum(cluster)

        if area < 5:
            continue

        # -----------------------------------
        # Edge-contact rejection
        # -----------------------------------

        top_contact = np.sum(
            cluster[0, :]
        )

        bottom_contact = np.sum(
            cluster[-1, :]
        )

        left_contact = np.sum(
            cluster[:, 0]
        )

        right_contact = np.sum(
            cluster[:, -1]
        )

        max_edge_contact = max(
            top_contact,
            bottom_contact,
            left_contact,
            right_contact
        )

        # Reject cluster only if it occupies
        # a large fraction of an image edge

        if max_edge_contact > 0.4 * max(
            roi_h,
            roi_w
        ):
            continue

        # -----------------------------------
        # Foot proximity check
        # -----------------------------------

        ys, xs = np.where(
            cluster
        )

        if len(xs) == 0:
            continue

        distances = np.sqrt(
            (xs - foot_x) ** 2 +
            (ys - foot_y) ** 2
        )

        if np.min(distances) < 12:

            player_mask[
                cluster
            ] = 1

    # -----------------------------------
    # Safety check
    # -----------------------------------

    if np.sum(player_mask) < 10:
        return None

    # -----------------------------------
    # HSV appearance
    # -----------------------------------

    hsv = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2HSV
    )

    hsv_pixels = hsv[
        player_mask > 0
    ]

    if len(hsv_pixels) == 0:
        return None

    median_hsv = np.median(
        hsv_pixels,
        axis=0
    )

    # -----------------------------------
    # Paint segmentation overlay
    # -----------------------------------

    mean_bgr = np.mean(
        roi[player_mask > 0],
        axis=0
    ).astype(np.uint8)

    segmentation_overlay[
        y1:y2,
        x1:x2
    ][player_mask > 0] = mean_bgr

    return median_hsv