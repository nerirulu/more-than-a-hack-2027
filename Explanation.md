# Technical Document — Possession Prediction Pipeline

**Match:** FC Barcelona vs Athletic Club, LaLiga EA Sports 2024-25, matchday 2
**Home:** Barcelona (`H`) **Away:** Athletic (`A`)
**Graded frames:** 158,857 of 199,319 total (≈106 minutes of play across two halves)

This document is the technical document required by [SUBMISSION.md](SUBMISSION.md) §3.

---

## 1. The Approach

### What we built

We built an end-to-end algorithmic pipeline that produces **one label per video frame** for the entire match (199,319 frames), where each label is one of:

| Label | Meaning |
|-------|---------|
| `H`   | Home team (Barcelona) in possession, ball alive |
| `A`   | Away team (Athletic) in possession, ball alive |
| `D`   | Dead ball — the ball is not in play |
| `X`   | No answer / abstention |

The pipeline runs through four Jupyter notebooks (§2) and outputs a `frame,label` CSV that satisfies `evaluation/validate_submission.py`.

### Why this rather than something else

The challenge provides **automatic, not corrected** tracking: the ball is missing on ~41% of graded frames, tracks are fragmented, there are identity switches, and there are **no player identifiers**. Hand labelling of the provided game is forbidden, and no labelled training set from this match is available. We chose a **geometry-and-event-driven rule pipeline** rather than a black-box ML possession classifier because:

1. **No training labels exist.** A supervised model would need ground-truth possession labels we are not allowed to use (§4).
2. **Restart structure is knowable.** Smart Tagging events (kick-offs, set pieces, players in the box) give strong signals about when the ball is dead and who will restart.
3. **Interpretability.** A rule-based pipeline lets us inspect exactly why a frame received a label, essential when the metric (spellF1) penalises transition errors heavily.

The one ML component is an **LSTM ball-position inference model** (notebook 2) to fill in the 41% of frames where the ball track is missing. It is a supporting signal, not the core classifier.

---

## 2. The Four Jupyter Notebooks

All notebooks live in `code/`. They are designed to run in order: 1 → 2 → 3 → 4.

### Notebook 1 — `1. EDA.ipynb` (Exploratory Data Analysis)

**Purpose:** Empirically identify the geometric thresholds and data characteristics that define ball possession and ball loss.

Key work:
- Loads the tactical ATD tracking data via `kloppy` (Metrica EPTS format). All 199,319 frames load successfully.
- Analyses ball tracking gaps: identifies blocks of consecutive frames where `ball_x`/`ball_y` is NaN (signal loss). Roughly 40–41% of frames have no ball detection.
- Ball-speed analysis: compares ball speed in normal frames vs. frames immediately before signal loss (Mann-Whitney U test).
- Produces a temporal "barcode" visualisation of ball presence vs absence across the match.

Output: an empirical understanding of how much ball data is missing, when it is missing, and how fast the ball typically moves — informing thresholds in later notebooks.

### Notebook 2 — `2. Ball Inference.ipynb` (Ball Position Inference)

**Purpose:** Predict the ball's (x, y) position on frames where automatic tracking lost it, using the positions and speeds of the 22 tracked players as context.

Key work:
- Loads all 199,319 frames of tactical tracking data.
- Defines `BallDataset`, a PyTorch `Dataset` that builds sliding windows of `SEQ_LEN = 25` frames (1 second at 25 fps). Each sample's feature vector is:
  - 22 players × 4 features (x, y, speed, distance to previous ball position) = 88 features
  - plus previous ball x, y = 2 features
  - total input dimension = 90
  - target = next frame's (ball_x, ball_y)
- Defines `BallPredictorLSTM`: an LSTM network (input 90, hidden 128, 2 layers, dropout 0.2, followed by Linear(128→64→2)) that predicts ball (x, y) directly.
- Trains for 15 epochs with Adam optimiser and MSE loss.
- Evaluates prediction error in metres (pitch scaling: ×105.3 for x, ×68.0 for y).

Output: a trained LSTM that can impute missing ball positions from player context, used as a supporting signal in the possession classifier.

### Notebook 3 — `3. Preparing Data.ipynb` (Data Preparation)

**Purpose:** Merge every available data source into one feature-ready dataframe that the final algorithm consumes.

Key work:
- Loads the Smart Tagging event stream: 638 instances with fields including `code` (event type), `Team`, `Type`, `Half`, timestamp, end_timestamp.
- Loads tactical tracking data via `kloppy` + `games.py`.
- Computes per-player speeds across frames.
- Merges tracking features and Smart Tagging events into a single dataframe.
- Outputs `dataframe_hackaton_final.csv`: **199,319 rows × 265 columns**, one row per video frame.

This is the single input file consumed by notebook 4.

### Notebook 4 — `4. Algorithm.ipynb` (The Possession Algorithm)

**Purpose:** Classify every frame as H / A / D / X and write the submission CSV.

Key work:
- **Match initialisation:** Loads `dataframe_hackaton_final.csv`, defines `MATCH_META` (per-half frame offsets and real playing halves), and computes `real_halves` — the actual frame ranges where play is live in each half.
- **Three-level possession classifier (applied per frame):**
  - **Level 1 — Set-piece / event forcing:** Smart Tagging events directly force certain labels (e.g., kick-off frames, players-in-the-box frames) to a specific team or to `D`.
  - **Level 2 — 75-frame geometric window:** For frames not forced by Level 1, the classifier looks at a 75-frame window (3 seconds at 25 fps), finds the team whose player is closest to the ball, and applies a 75% majority vote over the window.
  - **Level 3 — Attacking-half direction:** If still undecided, compares ball movement direction against each team's attacking half (determined from field-side detection at kick-off).
- **Ball validity:** Frames where the ball position is missing or below a speed threshold (`BALL_STOP = 0.1`) are treated as stopped/absent and handled by the dead-ball logic.
- **Output:** `df_results.to_csv("trial.csv", index=False)` with `frame` and `label` columns — 199,319 rows, frames 0–199318 in order, labels H/A/D/X.

Output: `trial.csv` — the candidate possession file.



---

## 3. Assets Used (and Dropped)

### Assets used

| Asset | Source | How used |
|-------|--------|----------|
| Tactical ATD tracking | `data/tactical/BAR-ATH_tactical_atd.txt` + `BAR-ATH_tactical_atd.xml` (via `kloppy`) | Primary input: per-frame ball + 46 track slots (position, speed) |
| Smart Tagging events | `data/tactical/BAR-ATH_tactical_smart_tagging.xml` | 638 events: set pieces, kick-offs, players in box, ball in final third, etc. — used to force Level-1 labels |
| Tactical homography | `data/tactical/BAR-ATH_tactical_homography.csv` | 100% valid; used for field-side / attacking-half determination |
| `games.py` | Repository starter code | Builds asset paths, handles `MTAH_DATA_DIR`, provides half-video blocks |
| `kloppy` | PyPI package | Reads Metrica EPTS tracking metadata + raw data |
| `dataframe_hackaton_final.csv` | Output of notebook 3 | Single merged feature table consumed by notebook 4 |

### Assets tried and dropped

| Asset | Why dropped |
|-------|-------------|
| **Broadcast recording** (`data/broadcast/`) | Not frame-synchronised to the tactical video; contains cuts, replays, and zooms; only 65.8% homography coverage; no mapping table to tactical frames. Too much alignment work for uncertain benefit. |
| **Supervised possession classifier (ML)** | No labelled training data exists for this match; using the public 5-minute ground truth for training is forbidden by the rules. |
| **Ball LSTM output as primary possession signal** | The LSTM predicts ball position, not possession. It was kept as a supporting imputation signal, not as the core classifier, because possession is better determined by player proximity and event structure than by imputed ball coordinates alone. |


---

## 4. Training Data

**No external labelled datasets were used.**

- **From the provided game:** Nothing. Per [RULES.md](RULES.md), no hand labels on the provided game may be used to train or validate a model, and the public 5-minute ground truth (`evaluation/gt/BAR-ATH_possession_gt_public.csv`) may be used only for pipeline validation — not for training, tuning, or as a training set.
- **From other games:** No labelled possession datasets from other games were used either. The entire pipeline is unsupervised / rule-based.
- **Who labelled what:** Not applicable — no human-produced possession labels were used at all.
- **Amount:** Zero labelled frames used for training.

The public 5-minute ground truth (frames 11185–18684, 17 possessions, H 3,454 · A 2,577 · D 1,469) was used solely to sanity-check that the pipeline's output format and broad label distribution are plausible.


---

## 5. Possession Rules Implemented

Our implementation follows the challenge's definition: **a change of possession occurs when a player from the opposing team gains controlled possession of the ball.** While play is live, one team is always in possession. The ground truth labels every live frame `H`, `A`, or `D`; nothing is smoothed, interpolated, or hand-adjusted.

### Label definitions (as used)

| Label | Rule |
|-------|------|
| `H` | Home team has controlled possession, ball alive |
| `A` | Away team has controlled possession, ball alive |
| `D` | Dead ball — ball not in play (restart, out of play, stoppage) |
| `X` | Abstention / ungraded frames |

### Where we drew the line

**Loose balls and challenges:** Possession stays with the last team that had it during contested or loose-ball phases. A simple touch without control, a deflection that doesn't intercept, or an aerial duel won without subsequent control does **not** change possession. This matches the challenge's "what does not count" list.

**Restarts:** At kick-off (start of half or after a goal) and set pieces, there is a clear possessor — the team taking the restart. Smart Tagging events identify these moments and Level 1 forces the correct label (or `D` while the ball is being put back into play). Possession can only change after play resumes and the opposing team gains controlled possession.

**Dead ball:** When the ball is dead, the answer is `D` regardless of which team will restart or touched it last. Dead-ball ownership is deliberately not asked for by the challenge.

**Classifier hierarchy:** Level 1 (events) → Level 2 (75-frame closest-team majority) → Level 3 (ball direction vs attacking half). If the ball is missing or stopped for an extended stretch, the frame leans toward `D` (dead ball) rather than inventing a possessor.

### Implementation note on `X`

`X` is used on ungraded frames (before kickoff, halftime, after the final whistle) where it costs nothing, and as a genuine abstention when the system cannot determine possession with confidence. Under `spellF1` (the ranking metric), an `X` is filled with the last label in that half before possession matching — so an `X` mid-possession costs little, but an `X` across a hand-over bleeds the old label into the new state and costs heavily. We therefore minimise `X` during live play and use it mainly where the ground truth itself is `X`.


---

## 6. Limitations

| Limitation | Impact | What we'd fix with more time |
|------------|--------|------------------------------|
| **Ball missing on ~41% of frames** | Forced reliance on player proximity and imputation; possession may be misassigned during long ball-loss stretches. | Improve the ball LSTM (more data, longer context, bidirectional) or use homography + player positioning to infer ball location more robustly. |
| **No player identifiers** | Track slots are anonymous (1–46); we cannot follow a specific player across identity switches. | Implement re-identification or use the provided homography to stabilise track identities. |
| **Fragmented tracks / identity switches** | Closest-player calculations can jump between players, causing noisy majority votes in Level 2. | Smooth track assignments temporally before computing proximity. |
| **Smart Tagging is inferred, not ground truth** | Event labels may be imperfect; forcing labels on wrong events propagates error. | Validate Smart Tagging against the public 5-minute ground truth and adjust confidence weighting per event type. |
| **Fixed thresholds** (75-frame window, 75% majority, `BALL_STOP = 0.1`) | Hand-picked, not learned; may not generalise across match phases. | Tune thresholds on the public ground truth (validation only) or learn them via a small search. |
| **Ball LSTM is position-only, not possession-aware** | Imputed ball position doesn't directly tell us who possesses it. | Extend the model to predict possession-relevant features (e.g., nearest player, controlling team) alongside position. |
| **No cross-validation on full match** | The private ground truth (158,857 graded frames) is unavailable, so we cannot measure true spellF1/macroF1 before submission. | Use the public 5 minutes for ablation and threshold tuning, accepting its limits. |


---

## 7. Reproduction — How to Run It

### Prerequisites

Python 3.9+ with the packages in `requirements.txt` / `requirements-starter.txt`, plus `kloppy`:

```bash
pip install -r requirements.txt
pip install kloppy
```

### Data

The match data is gated and **not in this repository** (it is gitignored under `data/`). Enrol at the More Than A Hack 2027 landing page, accept the terms, download the bundle, and unpack into `data/`:

```
data/
└─ tactical/
   ├─ BAR-ATH_tactical.mp4
   ├─ BAR-ATH_tactical_atd.xml
   ├─ BAR-ATH_tactical_atd.txt
   ├─ BAR-ATH_tactical_homography.csv
   └─ BAR-ATH_tactical_smart_tagging.xml
```

Set `MTAH_DATA_DIR` if the bundle lives elsewhere; `games.py` reads it automatically.

### Run the notebooks

Execute in order:

1. `code/1. EDA.ipynb` — explore thresholds and ball-gap characteristics (no output needed for the pipeline).
2. `code/2. Ball Inference.ipynb` — train the ball-position LSTM (optional supporting signal).
3. `code/3. Preparing Data.ipynb` — produces `dataframe_hackaton_final.csv` (199,319 × 265).
4. `code/4. Algorithm.ipynb` — loads the CSV, runs the three-level classifier, writes `trial.csv`.

### Validate before submitting

```bash
python evaluation/validate_submission.py trial.csv
```

This runs exactly the checks the official scorer runs (header, row count, frame ordering, label alphabet, no blanks, UTF-8, no index column). A file that passes is ready to upload.

### Output

`trial.csv` — 199,319 data rows, header `frame,label`, frames 0–199318 in order, labels H/A/D/X, UTF-8, no index column. Rename to `<team_name>.csv` for submission.


---

## 8. Submission Compliance (SUBMISSION.md)

### The possession file

Our `trial.csv` meets every strict requirement:

| Requirement | How we meet it |
|-------------|----------------|
| Header exactly `frame,label` | Written with `to_csv(index=False)`, header `frame,label` |
| Exactly 199,319 data rows | One row per tactical video frame |
| `frame` is `0 … 199318`, in order | Sequential, verified by `validate_submission.py` |
| `label` ∈ {`H`, `A`, `D`, `X`} | Stripped and upper-cased; only these four emitted |
| No blank cells | Every row has a label; `X` used for abstention |
| UTF-8, `\n` or `\r\n`, no index column | `to_csv(index=False)` default |

### The three deliverables

1. **The possession file** — `trial.csv` (rename to `<team_name>.csv`).
2. **A public code repository** — this repository contains source (`code/`, `starter/`, `evaluation/`), pinned dependencies (`requirements.txt`), and reproduction instructions (§7).
3. **A technical document** — this file (`Explanation.md`), to be renamed `<team_name>.pdf` or `<team_name>.md` for submission.

### How to submit

One Google Form (URL shared by email to registered participants) with four fields:

| Field | What to provide |
|-------|-----------------|
| Team Name | as stated in the registration form |
| Possession file | `<team_name>.csv` (~1.9 MB) |
| Public code repository URL | public URL of this repo |
| Short technical document | `<team_name>.md` or `<team_name>.pdf` (this file) |

**Deadline: November 17, 2026**

Practical notes from SUBMISSION.md:
- Name the CSV and technical document after your team — `<team_name>.csv` and `<team_name>.pdf`/`.md`.
- One response per team; the last response before the deadline is scored.
- There is no live leaderboard and no feedback — validate the exact file you upload with `validate_submission.py`.

### Rules compliance

- **Algorithmic labels only:** Every label in `trial.csv` is produced by the algorithm in notebook 4. No frame of the provided game was hand-labelled.
- **No private ground truth:** The public 5-minute ground truth was used only for format sanity-checking, never for training or tuning.
- **No human-produced data about the game:** No external match data or analyst labels were incorporated.

---

*End of technical document.*
