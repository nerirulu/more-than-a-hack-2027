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
N_FRAMES = 100

ROI_SIZE = 40

# controls how strongly distance is penalized
DISTANCE_SIGMA = 12

# =====================================================
# LOAD DATA
# =====================================================

homography = pd.read_csv(HOMO_PATH)

with open(ATD_PATH) as f:
    atd_lines = f.readlines()

cap = cv2.VideoCapture(VIDEO_PATH)

OUT_W = 1200
OUT_H = 600

fourcc = cv2.VideoWriter_fourcc(*"mp4v")


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

def find_gap_intervals():

    gap_intervals = []

    i = 1

    while i < len(atd_lines):

        prev_ball = get_ball(i - 1)
        curr_ball = get_ball(i)

        # visible -> missing
        if prev_ball is not None and curr_ball is None:

            start_missing = i

            j = i

            while (
                j < len(atd_lines)
                and get_ball(j) is None
            ):
                j += 1

            end_missing = j - 1

            length = (
                end_missing -
                start_missing +
                1
            )

            gap_intervals.append(
                {
                    "start_missing": start_missing,
                    "end_missing": end_missing,
                    "length": length,
                    "last_visible": start_missing - 1
                }
            )

            i = j

        else:
            i += 1

    return gap_intervals

gap_intervals = find_gap_intervals()

HALVES_RANGES = {
    1: (1713242, 1792708),
    2: (1812866, 1892255)
}

HALVES_OFFSETS = {
    1: 1702057,
    2: 1702058
}

half1_start = HALVES_RANGES[1][0] - HALVES_OFFSETS[1]
half1_end   = HALVES_RANGES[1][1] - HALVES_OFFSETS[1]

half2_start = HALVES_RANGES[2][0] - HALVES_OFFSETS[2]
half2_end   = HALVES_RANGES[2][1] - HALVES_OFFSETS[2]

valid_gap_intervals = []

for gap in gap_intervals:

    frame = gap["last_visible"]

    in_half1 = (
        half1_start <= frame <= half1_end
    )

    in_half2 = (
        half2_start <= frame <= half2_end
    )

    if in_half1 or in_half2:
        valid_gap_intervals.append(gap)

gap_intervals = valid_gap_intervals

gap_intervals = [
    g for g in gap_intervals
    if g["length"] >= 10
]

print("\nALL GAPS\n")

for k, gap in enumerate(gap_intervals):

    print(
        f"{k:3d} | "
        f"start={gap['start_missing']} "
        f"end={gap['end_missing']} "
        f"length={gap['length']}"
    )

# =====================================================
# MAIN LOOP
# =====================================================

for gap_id, gap in enumerate(gap_intervals):

    print(
        f"\nProcessing gap {gap_id}"
        f" start={gap['start_missing']}"
        f" end={gap['end_missing']}"
        f" length={gap['length']}"
    )

    OUTPUT_VIDEO = (
        f"gap_{gap_id:03d}"
        f"_len_{gap['length']:03d}.mp4"
    )

    tracked_px = None
    tracked_py = None

    START_FRAME = gap["last_visible"]

    N_FRAMES = gap["length"] + 5

    ball0 = get_ball(START_FRAME)
    ball1 = get_ball(START_FRAME - 1)

    if ball0 is None or ball1 is None:
        continue

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        fourcc,
        10,
        (OUT_W, OUT_H)
    )

    vx = ball0[0] - ball1[0]
    vy = ball0[1] - ball1[1]

    pred_x = ball0[0]
    pred_y = ball0[1]

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
            95
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

        else:

            tracked_x = centre_x
            tracked_y = centre_y

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

        # ==========================================
        # BUILD DEBUG MOSAIC
        # ==========================================

        roi_show = roi_rgb.copy()

        sat_show = cv2.cvtColor(
            S_channel,
            cv2.COLOR_GRAY2RGB
        )

        resp_show = cv2.cvtColor(
            ball_response,
            cv2.COLOR_GRAY2RGB
        )

        blur_show = cv2.cvtColor(
            ball_response_blur,
            cv2.COLOR_GRAY2RGB
        )

        mask_show = (
            candidate_mask.astype(np.uint8) * 255
        )

        mask_show = cv2.cvtColor(
            mask_show,
            cv2.COLOR_GRAY2RGB
        )

        win_show = candidate_img.copy()

        # --------------------------------
        # resize everything
        # --------------------------------

        CELL_W = 400
        CELL_H = 300

        roi_show = cv2.resize(
            roi_show,
            (CELL_W, CELL_H)
        )

        sat_show = cv2.resize(
            sat_show,
            (CELL_W, CELL_H)
        )

        resp_show = cv2.resize(
            resp_show,
            (CELL_W, CELL_H)
        )

        blur_show = cv2.resize(
            blur_show,
            (CELL_W, CELL_H)
        )

        mask_show = cv2.resize(
            mask_show,
            (CELL_W, CELL_H)
        )

        win_show = cv2.resize(
            win_show,
            (CELL_W, CELL_H)
        )

        # --------------------------------
        # titles
        # --------------------------------

        cv2.putText(
            roi_show,
            "ROI",
            (10,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,0,0),
            2
        )

        cv2.putText(
            sat_show,
            "SATURATION",
            (10,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,0,0),
            2
        )

        cv2.putText(
            resp_show,
            "255-S",
            (10,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,0,0),
            2
        )

        cv2.putText(
            blur_show,
            "GAUSSIAN",
            (10,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,0,0),
            2
        )

        cv2.putText(
            mask_show,
            "MASK",
            (10,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,0,0),
            2
        )

        cv2.putText(
            win_show,
            f"WINNER score={best_score:.1f}",
            (10,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,0,0),
            2
        )

        # --------------------------------
        # grid
        # --------------------------------

        top = np.hstack([
            roi_show,
            sat_show,
            resp_show
        ])

        bottom = np.hstack([
            blur_show,
            mask_show,
            win_show
        ])

        debug_frame = np.vstack([
            top,
            bottom
        ])

        # --------------------------------
        # frame info
        # --------------------------------

        cv2.putText(
            debug_frame,
            f"Frame {frame_idx}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0,255,0),
            3
        )

        cv2.putText(
            debug_frame,
            f"Gap {gap_id} | len={gap['length']}",
            (20,80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,255,0),
            2
        )

        writer.write(
            cv2.cvtColor(
                debug_frame,
                cv2.COLOR_RGB2BGR
            )
        )

    writer.release()

    print("\nSaved:")
    print(OUTPUT_VIDEO)

cap.release()