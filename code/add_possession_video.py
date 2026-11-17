import cv2
import pandas as pd
from tqdm import tqdm

# ==================================
# FILES
# ==================================

VIDEO_PATH = "full_debug_match.mp4"
CSV_PATH = "submission.csv"

OUTPUT_PATH = "full_debug_match_possession.mp4"

# ==================================
# LOAD POSSESSION
# ==================================

print("Loading possession labels...")

possession_df = pd.read_csv(CSV_PATH)

frame_to_label = dict(
    zip(
        possession_df["frame"],
        possession_df["label"]
    )
)

print(
    f"Loaded {len(possession_df):,} labels"
)

# ==================================
# VIDEO
# ==================================

print("Opening video...")

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

print(f"FPS          : {fps}")
print(f"Resolution   : {width} x {height}")
print(f"Total frames : {total_frames:,}")

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# ==================================
# PROCESS
# ==================================

print("\nRendering possession video...\n")

pbar = tqdm(
    total=total_frames,
    desc="Video",
    unit="frames",
    dynamic_ncols=True
)

frame_idx = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    label = frame_to_label.get(
        frame_idx,
        "X"
    )

    # ----------------------------------
    # COLORS + TEXT
    # ----------------------------------

    if label == "H":

        text = "POSSESSION: HOME"
        color = (255, 0, 0)

    elif label == "A":

        text = "POSSESSION: AWAY"
        color = (0, 0, 255)

    elif label == "D":

        text = "POSSESSION: DEAD BALL"
        color = (0, 255, 255)

    else:

        text = "POSSESSION: X"
        color = (180, 180, 180)

    # ----------------------------------
    # TEXT SIZE
    # ----------------------------------

    (tw, th), _ = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        2
    )

    x = width - tw - 30
    y = height - 40

    # ----------------------------------
    # BACKGROUND
    # ----------------------------------

    cv2.rectangle(
        frame,
        (x - 15, y - th - 15),
        (x + tw + 15, y + 10),
        (30, 30, 30),
        -1
    )

    # ----------------------------------
    # BORDER
    # ----------------------------------

    cv2.rectangle(
        frame,
        (x - 15, y - th - 15),
        (x + tw + 15, y + 10),
        color,
        2
    )

    # ----------------------------------
    # TEXT
    # ----------------------------------

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
        cv2.LINE_AA
    )

    # ----------------------------------
    # WRITE FRAME
    # ----------------------------------

    writer.write(frame)

    frame_idx += 1
    pbar.update(1)

    # Optional extra info every 5000 frames

    if frame_idx % 5000 == 0:

        pbar.set_postfix(
            frame=frame_idx,
            label=label
        )

# ==================================
# FINISH
# ==================================

pbar.close()

cap.release()
writer.release()

print("\nDone!")
print(
    f"Saved video to: {OUTPUT_PATH}"
)