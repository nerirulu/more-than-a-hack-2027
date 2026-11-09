import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

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

    return feature

# =====================================================
# PARAMETERS
# =====================================================

VIDEO_PATH = "../data/tactical/BAR-ATH_tactical.mp4"
ATD_PATH = "../data/tactical/BAR-ATH_tactical_atd.txt"
HOMO_PATH = "../data/tactical/BAR-ATH_tactical_homography.csv"

FRAME = 11550

PEAK_RADIUS = 30

# =====================================================
# LOAD FRAME
# =====================================================

cap = cv2.VideoCapture(VIDEO_PATH)

cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    FRAME
)

ret, frame = cap.read()

cap.release()

if not ret:
    raise RuntimeError(
        "Could not read frame"
    )

# =====================================================
# LOAD HOMOGRAPHY
# =====================================================

homography = pd.read_csv(
    HOMO_PATH
)

row = homography.iloc[
    FRAME
]

H = np.array([
    [row["h00"], row["h01"], row["h02"]],
    [row["h10"], row["h11"], row["h12"]],
    [row["h20"], row["h21"], row["h22"]],
])

# =====================================================
# LOAD ATD
# =====================================================

with open(ATD_PATH) as f:

    atd_lines = f.readlines()

line = atd_lines[
    FRAME
].strip()

players = line.split(":")[1].split(";")

# =====================================================
# PROJECT PLAYERS
# =====================================================

projected_players = []

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

        if (
            px >= 0 and
            px < frame.shape[1] and
            py >= 0 and
            py < frame.shape[0]
        ):

            projected_players.append(
                (int(px), int(py))
            )

    except:
        pass

print(
    f"Players found: {len(projected_players)}"
)

# =====================================================
# BUILD FEATURES
# =====================================================

all_features = []
all_rois = []

for idx, (px, py) in enumerate(
    projected_players
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
        continue

    feature = extract_player_feature(
        frame,
        px,
        py,
        PEAK_RADIUS
    )

    if feature is None:

        print(
            f"Player {idx} rejected"
        )

        continue

    roi_rgb = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2RGB
    )

    all_features.append(
        feature
    )

    all_rois.append(
        (
            idx,
            roi_rgb
        )
    )

# =====================================================
# CLUSTER INTO 2 TEAMS
# =====================================================

all_features = np.array(
    all_features
)

kmeans = KMeans(
    n_clusters=2,
    random_state=0,
    n_init=20
)

labels = kmeans.fit_predict(
    all_features
)

# =====================================================
# SHOW RESULTS
# =====================================================

num_players = len(
    all_rois
)

cols = 5

rows = int(
    np.ceil(
        num_players / cols
    )
)

fig, axes = plt.subplots(
    rows,
    cols,
    figsize=(
        cols * 3,
        rows * 3
    )
)

axes = np.array(axes).reshape(-1)

for ax in axes:
    ax.axis("off")

for ax, ((player_idx,
          roi_rgb),
         team_label) in zip(
             axes,
             zip(
                 all_rois,
                 labels
             )
         ):

    ax.imshow(
        roi_rgb
    )

    title_color = (
        "blue"
        if team_label == 0
        else "red"
    )

    ax.set_title(
        f"P{player_idx}\nTeam {team_label}",
        color=title_color,
        fontsize=10
    )

plt.tight_layout()
plt.show()

# =====================================================
# PRINT ASSIGNMENTS
# =====================================================

print("\n===================")
print("TEAM ASSIGNMENTS")
print("===================")

for (
    player_idx,
    _
), label in zip(
    all_rois,
    labels
):

    print(
        f"Player {player_idx}: Team {label}"
    )