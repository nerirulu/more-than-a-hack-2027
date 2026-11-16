import numpy as np
import pandas as pd

df = pd.read_csv("data/dataframe_hackaton_final.csv", low_memory=False)
print(df.shape)
print(df[["period_id", "timestamp", "frame_id"]].head(3))
print("frame_id range:", df["frame_id"].min(), "->", df["frame_id"].max())


MATCH_META = {"offsets": {1: 1702057, 2: 1702058},
              "halves": {1: (1713242, 1792708), 2: (1812866, 1892255)},
              "kickoffs": {1: {"minute": 7, "second": 27, "frame": 11},
                           2: {"minute": 75, "second": 49, "frame": 16}}}
k = MATCH_META["kickoffs"]
real_halves = {h: ((k[h]["minute"] * 60 + k[h]["second"]) * 25 + k[h]["frame"],
                   MATCH_META["halves"][h][1] - MATCH_META["offsets"][h]) for h in [1, 2]}
print(real_halves)
for h, (s, e) in real_halves.items():
    sub = df[(df["frame_id"] >= s) & (df["frame_id"] <= e)]
    print(f"half {h}: {len(sub)} rows (expected {e - s + 1}), period_id values: {sub['period_id'].unique()}")

# %% 1.3 What do the coordinates look like?
cols = ["ball_x", "ball_y", "1_x", "1_y", "30_x", "30_y"]
print(df[cols].describe().loc[["count", "min", "max"]])

# %% 1.4 Tracking coverage in play
play = pd.concat([df[(df["frame_id"] >= s) & (df["frame_id"] <= e)] for s, e in real_halves.values()])
print("ball tracked:", f"{play['ball_x'].notna().mean():.1%}")
home_n = play[[f"{i}_x" for i in range(1, 25)]].notna().sum(axis=1)
away_n = play[[f"{i}_x" for i in range(25, 47)]].notna().sum(axis=1)
print("home players tracked per frame:\n", home_n.describe()[["min", "50%", "max"]])
print("away players tracked per frame:\n", away_n.describe()[["min", "50%", "max"]])

#%%
print((home_n > 11).mean(), (away_n > 11).mean())     # share of frames with more than 11
print(home_n.value_counts().sort_index())
print(away_n.value_counts().sort_index())
print("frames with 0 home players:", (home_n == 0).sum(), "| ball and no players:", ((home_n + away_n == 0) & play['ball_x'].notna()).sum())

# one example frame with the most players, and who is there
f = home_n.idxmax()
row = play.loc[f]
print("frame_id", row["frame_id"], "-> tracked home ids:", [i for i in range(1, 25) if pd.notna(row[f"{i}_x"])])
# %%
