import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

# =====================================================
# PARAMETERS
# =====================================================

VIDEO_PATH = "../data/tactical/BAR-ATH_tactical.mp4"
ATD_PATH = "../data/tactical/BAR-ATH_tactical_atd.txt"
HOMO_PATH = "../data/tactical/BAR-ATH_tactical_homography.csv"

START_FRAME = 11150
N_FRAMES = 1000

MAX_DISTANCE = 0.03

TEAM_CHECK_INTERVAL = 10

WARMUP_FRAMES = 100

TEAM_SWITCH_VOTES = 10

MAX_INVALID_CHECKS = 10

BALL_ROI_SIZE = 40

BALL_MAX_AREA = 150

BALL_MIN_AREA = 3

BALL_MAX_JUMP_PX = 15

OUTPUT_VIDEO = "team_tracking_debug.mp4"

# =====================================================
# TEAM COLORS
# =====================================================

TEAM_0 = (255, 0, 0)       # blue
TEAM_1 = (0, 0, 255)       # red

# =====================================================
# LOAD HOMOGRAPHY
# =====================================================

homography = pd.read_csv(HOMO_PATH)

# =====================================================
# LOAD ATD
# =====================================================

with open(ATD_PATH) as f:
    atd_lines = f.readlines()

# =====================================================
# VIDEO
# =====================================================

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# =====================================================
# TRACKING STRUCTURES
# =====================================================

tracks = {}

next_track_id = 0

team_kmeans = None
team_classifier_ready = False

feature_bank = []

ball_track = {

    "position": None,

    "velocity": np.array(
        [0.0, 0.0]
    ),

    "missing_frames": 0,

    "visible": False
}

MAX_BALL_GAP = 25

# =====================================================
# HELPERS
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

def detect_ball_candidate(
    frame,
    predicted_px,
    predicted_py
):

    x1 = max(
        0,
        predicted_px - BALL_ROI_SIZE
    )

    x2 = min(
        frame.shape[1],
        predicted_px + BALL_ROI_SIZE
    )

    y1 = max(
        0,
        predicted_py - BALL_ROI_SIZE
    )

    y2 = min(
        frame.shape[0],
        predicted_py + BALL_ROI_SIZE
    )

    roi = frame[
        y1:y2,
        x1:x2
    ]

    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2HSV
    )

    S_channel = hsv[:,:,1]

    response = (
        255 - S_channel
    ).astype(np.uint8)

    response = cv2.GaussianBlur(
        response,
        (9,9),
        0
    )

    candidate_mask = (
        response >
        np.percentile(
            response,
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

        if area < BALL_MIN_AREA:
            continue

        if area > BALL_MAX_AREA:
            continue

        cx, cy = centroids[i]

        jump = np.sqrt(
            (cx - centre_x)**2 +
            (cy - centre_y)**2
        )

        if jump > BALL_MAX_JUMP_PX:
            continue

        mask = (
            labels == i
        )

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

        peak = response[
            mask
        ].max()

        score = (
            peak
            *
            circularity
            /
            np.sqrt(area)
        )

        if score > best_score:

            best_score = score

            best_centroid = (
                int(cx),
                int(cy)
            )

    if best_centroid is None:
        return None

    return np.array([
        x1 + best_centroid[0],
        y1 + best_centroid[1]
    ])

def get_players_and_ball(frame_number):

    line = atd_lines[frame_number].strip()

    parts = line.split(":")

    if len(parts) < 3:
        return [], None

    players = parts[1].split(";")

    positions = []

    for player in players:

        if player.startswith("NaN"):
            continue

        try:

            x, y, speed = player.split(",")

            positions.append(
                (
                    float(x),
                    float(y)
                )
            )

        except:
            pass

    ball = None

    try:

        bx, by, bspeed = parts[2].split(",")

        if (
            bx != "NaN"
            and by != "NaN"
        ):

            ball = (
                float(bx),
                float(by)
            )

    except:
        pass

    return positions, ball

def extract_player_feature(
    frame,
    px,
    py,
    peak_radius=30
):

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
        py - int(
            player_height * 1.8
        )
    )

    y2 = min(
        frame.shape[0],
        py + 5
    )

    roi = frame[
        y1:y2,
        x1:x2
    ]

    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2HSV
    )

    H_hist = cv2.calcHist(
        [hsv],
        [0],
        None,
        [180],
        [0,180]
    ).flatten()

    S_hist = cv2.calcHist(
        [hsv],
        [1],
        None,
        [256],
        [0,256]
    ).flatten()

    V_hist = cv2.calcHist(
        [hsv],
        [2],
        None,
        [256],
        [0,256]
    ).flatten()

    h_peak = np.argmax(
        H_hist
    )

    H_hist_removed = H_hist.copy()

    H_hist_removed[
        max(
            0,
            h_peak - peak_radius
        ):
        min(
            len(H_hist_removed),
            h_peak + peak_radius + 1
        )
    ] = 0

    remaining_hue_pixels = np.sum(
        H_hist_removed
    )

    # Reject bad ROI

    if remaining_hue_pixels < 10:
        return None

    s_peak = np.argmax(
        S_hist
    )

    S_hist[
        max(
            0,
            s_peak - peak_radius
        ):
        min(
            len(S_hist),
            s_peak + peak_radius + 1
        )
    ] = 0

    v_peak = np.argmax(
        V_hist
    )

    V_hist[
        max(
            0,
            v_peak - peak_radius
        ):
        min(
            len(V_hist),
            v_peak + peak_radius + 1
        )
    ] = 0

    H_hist = H_hist_removed / (
        H_hist_removed.sum() + 1e-6
    )

    S_hist = S_hist / (
        S_hist.sum() + 1e-6
    )

    V_hist = V_hist / (
        V_hist.sum() + 1e-6
    )

    feature = np.concatenate(
        [
            5 * H_hist,
            3 * S_hist,
            1 * V_hist
        ]
    )

    return feature.astype(np.float64)

# =====================================================
# MAIN LOOP
# =====================================================

for frame_idx in range(
    START_FRAME,
    START_FRAME + N_FRAMES
):

    print(
        f"{frame_idx}/{START_FRAME + N_FRAMES}"
    )

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_idx
    )

    ret, frame = cap.read()

    if not ret:
        break

    segmentation_overlay = np.zeros_like(frame)

    row = homography.iloc[frame_idx]

    H = np.array([
        [row["h00"], row["h01"], row["h02"]],
        [row["h10"], row["h11"], row["h12"]],
        [row["h20"], row["h21"], row["h22"]],
    ])

    detections, ball = (
        get_players_and_ball(
            frame_idx
        )
    )

    detections = np.array(
        detections,
        dtype=np.float32
    )

    # =================================================
    # BALL TRACKING
    # =================================================

    if ball is not None:

        ball = np.array(
            ball,
            dtype=np.float64
        )

        if ball_track["position"] is not None:

            ball_track["velocity"] = (
                ball
                -
                ball_track["position"]
            )

        ball_track["position"] = ball

        ball_track["missing_frames"] = 0

        ball_track["visible"] = True

    else:

        if ball_track["position"] is not None:

            predicted_ball = (
                ball_track["position"]
                +
                ball_track["velocity"]
            )

            pred_x, pred_y = predicted_ball

            pred_px, pred_py = project(
                H,
                pred_x,
                pred_y
            )

            candidate_px_py = (
                detect_ball_candidate(
                    frame,
                    pred_px,
                    pred_py
                )
            )

            if candidate_px_py is not None:

                candidate_px = (
                    candidate_px_py[0]
                )

                candidate_py = (
                    candidate_px_py[1]
                )

                invH = np.linalg.inv(H)

                denom = (
                    invH[2,0] * candidate_px +
                    invH[2,1] * candidate_py +
                    invH[2,2]
                )

                candidate_x = (
                    invH[0,0] * candidate_px +
                    invH[0,1] * candidate_py +
                    invH[0,2]
                ) / denom

                candidate_y = (
                    invH[1,0] * candidate_px +
                    invH[1,1] * candidate_py +
                    invH[1,2]
                ) / denom

                candidate_position = np.array(
                    [
                        candidate_x,
                        candidate_y
                    ]
                )

                ball_track["velocity"] = (
                    candidate_position
                    -
                    ball_track["position"]
                )

                ball_track["position"] = (
                    candidate_position
                )

            else:

                ball_track["position"] = (
                    predicted_ball
                )

            ball_track["missing_frames"] += 1

            ball_track["visible"] = False

            if (
                ball_track["missing_frames"]
                > MAX_BALL_GAP
            ):

                ball_track["position"] = None

    do_team_check = (
        (frame_idx - START_FRAME)
        % TEAM_CHECK_INTERVAL
        == 0
    )

    if len(tracks) == 0:

        for det in detections:

            tracks[next_track_id] = {

                "position": det,

                "feature": None,

                "team": -1,

                "team_votes": [0, 0],

                "invalid_checks": 0
            }

            next_track_id += 1

        continue

    # =================================================
    # TRACKING UPDATE
    # =================================================

    else:

        track_ids = list(
            tracks.keys()
        )

        track_positions = np.array([
            tracks[t]["position"]
            for t in track_ids
        ])

        cost = np.linalg.norm(
            track_positions[:, None, :]
            - detections[None, :, :],
            axis=2
        )

        row_ind, col_ind = (
            linear_sum_assignment(
                cost
            )
        )

        new_tracks = {}

        matched_dets = set()

        for r, c in zip(
            row_ind,
            col_ind
        ):

            distance = cost[r, c]

            if distance > MAX_DISTANCE:
                continue

            track_id = track_ids[r]

            det = detections[c]

            x, y = det

            px, py = project(
                H,
                x,
                y
            )

            feature = tracks[track_id]["feature"]

            if do_team_check:

                new_feature = extract_player_feature(
                    frame,
                    px,
                    py
                )

                if new_feature is not None:
                    feature = new_feature

            if (
                feature is not None
                and not team_classifier_ready
            ):
                feature_bank.append(feature)

            team = tracks[
                track_id
            ]["team"]

            team_votes = tracks[
                track_id
            ]["team_votes"]

            invalid_checks = tracks[
                track_id
            ]["invalid_checks"]

            if (
                team_classifier_ready
                and do_team_check
            ):

                if feature is None:

                    invalid_checks += 1

                else:

                    invalid_checks = 0

                    predicted_team = int(
                        team_kmeans.predict(
                            np.array(
                                [feature],
                                dtype=np.float64
                            )
                        )[0]
                    )

                    team_votes[
                        predicted_team
                    ] += 1

                    if (
                        team_votes[0]
                        >= TEAM_SWITCH_VOTES
                    ):

                        team = 0

                        team_votes = [0, 0]

                    elif (
                        team_votes[1]
                        >= TEAM_SWITCH_VOTES
                    ):

                        team = 1

                        team_votes = [0, 0]

            if invalid_checks >= MAX_INVALID_CHECKS:

                continue

            new_tracks[
                track_id
            ] = {
                "position": det,

                "feature": feature,

                "team": team,

                "team_votes": team_votes,

                "invalid_checks":
                    invalid_checks
            }

            matched_dets.add(c)

        for i, det in enumerate(
            detections
        ):

            if i in matched_dets:
                continue

            x, y = det

            px, py = project(
                H,
                x,
                y
            )

            feature = extract_player_feature(
                frame,
                px,
                py
            )

            if (
                feature is not None
                and not team_classifier_ready
            ):
                feature_bank.append(
                    feature
                )

            team = -1

            if (
                team_classifier_ready
                and feature is not None
            ):

                team = int(
                    team_kmeans.predict(
                        np.array(
                            [feature],
                            dtype=np.float64
                        )
                    )[0]
                )

            new_tracks[
                next_track_id
            ] = {

                "position": det,

                "feature": feature,

                "team": team,

                "team_votes": [0, 0],

                "invalid_checks": 0
            }

            next_track_id += 1

        tracks = new_tracks

    if (
        not team_classifier_ready
        and frame_idx >= START_FRAME + WARMUP_FRAMES
    ):

        if len(feature_bank) > 20:

            print(
                f"Training team classifier with "
                f"{len(feature_bank)} samples"
            )

            team_kmeans = KMeans(
                n_clusters=2,
                random_state=0,
                n_init=20
            )

            team_kmeans.fit(
                np.array(
                    feature_bank,
                    dtype=np.float64
                )
            )

            team_classifier_ready = True

            for track_id in tracks:

                feature = tracks[track_id]["feature"]

                if feature is None:
                    continue

                tracks[track_id]["team"] = int(
                    team_kmeans.predict(
                        np.array(
                            [feature],
                            dtype=np.float64
                        )
                    )[0]
                )

            print(
                "Team classifier ready"
            )

    # =================================================
    # DRAW SEGMENTATION
    # =================================================

    overlay = frame.copy()

    if ball_track["position"] is not None:

        bx, by = ball_track["position"]

        ball_px, ball_py = project(
            H,
            bx,
            by
        )

        if (
            0 <= ball_px < width
            and
            0 <= ball_py < height
        ):

            if ball_track["visible"]:

                ball_color = (
                    255,
                    255,
                    255
                )

            else:

                ball_color = (
                    0,
                    255,
                    255
                )

            cv2.circle(
                overlay,
                (
                    ball_px,
                    ball_py
                ),
                10,
                ball_color,
                -1
            )

            cv2.circle(
                overlay,
                (
                    ball_px,
                    ball_py
                ),
                12,
                (0,0,0),
                2
            )

    for track_id, track in tracks.items():

        x, y = track["position"]

        px, py = project(
            H,
            x,
            y
        )

        if (
            px < 0 or px >= width
            or py < 0 or py >= height
        ):
            continue

        team = track["team"]

        if team == 0:

            color = TEAM_0

        elif team == 1:

            color = TEAM_1

        else:

            color = (0, 255, 255)

        cv2.circle(
            overlay,
            (px, py),
            6,
            color,
            -1
        )

        cv2.putText(
            overlay,
            str(track_id),
            (px + 8, py),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1
        )

    writer.write(overlay)

# =====================================================
# FINISH
# =====================================================

cap.release()
writer.release()

print()
print("Saved:")
print(OUTPUT_VIDEO)