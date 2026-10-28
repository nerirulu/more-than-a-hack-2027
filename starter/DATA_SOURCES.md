# Data sources

All data comes from Metrica Sports' computer-vision pipeline, which processes
two camera feeds of the same LaLiga match (Barcelona vs Athletic Club,
2024-25, matchday 2).

## The two video feeds

| Feed | File | Role |
|---|---|---|
| Tactical | `data/tactical/BAR-ATH_tactical.mp4` | The frame-index authority: submission row `i` is video frame `i`. 199,319 frames at 25 fps |
| Broadcast | `data/broadcast/BAR-ATH_broadcast.mp4` | Optional extra. Own timeline (164,488 frames), cuts/replays, not frame-aligned to tactical |

## The four data assets per feed

| Asset | Tactical | Broadcast |
|---|---|---|
| Video | 199,319 frames | 164,488 frames |
| ATD (tracking) | 199,319 records | 164,488 records |
| Homography | 100% valid | 65.8% valid |
| Smart Tagging | 638 phases | — |

## How the assets relate to video frames

| Source | How to get the video frame |
|---|---|
| Submission row `i` | == video frame `i` |
| ATD record `n` | == video frame `n - 1` |
| Smart Tagging `seconds` | == video frame `seconds * 25` |
| Homography row `f` | == video frame `f` (already a submission row) |

## Frame alignment details

- Game: `BAR-ATH` (Barcelona vs Athletic Club)
- 1st half graded frames: 11,185–90,651
- 2nd half graded frames: 110,808–190,197
- Graded total: **158,857 frames**
- Ungraded total: **40,462 frames**

## ATD format (raw text file)

Each line is one frame:

```
<frame_id>:<slot1>;<slot2>;...;<slot46>:<ball>
```

Each slot is `x,y,speed` in normalised 0-1 coordinates (NaN = missing).
The ball is the final field. Example:

```
1:0.33411,0.90565,0.88686;NaN,NaN,NaN;...:0.10255,0.48070,NaN:0.67257,0.53268,1.36246
```

## How to load the ATD in Python

```python
from kloppy import metrica
dataset = metrica.load_tracking_epts(meta_data=paths["atd_meta"], raw_data=paths["atd_raw"])
```

For a streaming, numpy-only reader, see `starter/visualize_tracking.py::read_atd_window()`.