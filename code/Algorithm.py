import numpy as np
import pandas as pd

# ==========================================
# CONFIGURACIÓN DE VARIABLES
# ==========================================

# Variables generales
FPS = 25               # Frames por segundo
VIDEO_FRAMES = 199319  # Frames totales del video
PITCH_CENTER_X = 0.5   # Cambiar a 52.5 si el campo va de 0 a 105
PITCH_CENTER_Y = 0.5   # Cambiar a 34.0 si el campo va de 0 a 68
PITCH_LENGTH = 105.0   # Longitud del campo en metros
PITCH_WIDTH = 68.0     # Anchura del campo en metros

# Frames de las partes
MATCH_META = {"offsets": {1: 1702057, 2: 1702058},
              "halves": {1: (1713242, 1792708), 2: (1812866, 1892255)},
              "kickoffs": {1: {"minute": 7, "second": 27, "frame": 11}, 2: {"minute": 75, "second": 49, "frame": 16}}}
real_halves = {half: (
        (MATCH_META["kickoffs"][half]["minute"] * 60 + MATCH_META["kickoffs"][half]["second"]) * 25 + MATCH_META["kickoffs"][half]["frame"],
        MATCH_META["halves"][half][1] - MATCH_META["offsets"][half])
    for half in [1, 2]}

# Etiquetas y rangos de jugadores por equipo
HOME_TAG = "H"   # Cambia a "H" o el string exacto que os pidan en el submission
AWAY_TAG = "A"   # Cambia a "A" o el string exacto que os pidan en el submission
HOME_IDS = list(range(1, 25))   # Jugadores del 1 al 24
AWAY_IDS = list(range(25, 47))  # Jugadores del 25 al 46

# Variables threshoold
TH_BALL_SPEED = 0.1




# ==========================================
# PASO 0: PONER LAS 'X' Y SEPARAR PARTES
# ==========================================
def init_frames_with_X(halves_dict=real_halves, n_video_frames=VIDEO_FRAMES):
    """
    Inicializa el DataFrame de predicciones para todo el vídeo.
    Etiqueta con 'X' los tramos fuera de juego (pre-partido, descanso y post-partido)
    y deja en None los frames correspondientes a la 1ª y 2ª parte.

    Args:
        halves_dict (dict): Diccionario {1: (start_1, end_1), 2: (start_2, end_2)} en frames reales.
        n_video_frames (int): Número total de frames del vídeo.

    Returns:
        pd.DataFrame: DataFrame con columnas ['frame', 'label'].
    """

    # 1. Crear el DataFrame con las dos columnas: 'frame' (0 a n_video_frames - 1) y 'label'
    # Inicializamos todo con "X" por defecto
    df_results = pd.DataFrame({
        "frame": range(n_video_frames),
        "label": "X"
    })
    
    # 2. Dejar vacíos (None) los frames que sí corresponden a la 1ª y 2ª parte
    for half, (start_frame, end_frame) in halves_dict.items():
        df_results.loc[start_frame:end_frame, "label"] = None
        
    return df_results

# ==========================================
# PASO 1: FUNCIÓN DE KICK OFF (START)
# ==========================================

def kick_off(df, start_frame, max_wait_frames=500):
    """
    Analiza el saque de centro al inicio de una parte para determinar hacia dónde
    ataca cada equipo, quién empieza con la posesión y en qué frame se pone el balón en juego.

    Args:
        df (pd.DataFrame): DataFrame con los datos de tracking y velocidades.
        start_frame (int): Frame real en el que comienza la parte.
        max_wait_frames (int, optional): Límite de frames a explorar hasta el golpeo inicial.

    Returns:
        tuple:
            - attack_dir (dict): Dirección de ataque por equipo ({'H': 1/-1, 'A': -1/1}).
            - starting_team (str): Etiqueta del equipo que realiza el saque ('H' o 'A').
            - play_start_frame (int): Frame exacto en el que el balón empieza a moverse.
    """

    # 0. Localizar el frame de inicio en el df
    row_mask = df["frame_id"] == start_frame
    if row_mask.any():
        row = df.loc[row_mask].iloc[0]
    else:
        # Por si falta algún frame exacto, cogemos el primero disponible a partir de start_frame
        row = df.loc[df["frame_id"] >= start_frame].iloc[0]
    # Columnas X de cada equipo
    home_x_cols = [f"{i}_x" for i in HOME_IDS if f"{i}_x" in df.columns]
    away_x_cols = [f"{i}_x" for i in AWAY_IDS if f"{i}_x" in df.columns]


    # 1. Detectar donde ataca cada equipo (Posición jugadores a banda y banda)
    home_median_x = row[home_x_cols].median()
    away_median_x = row[away_x_cols].median()
    # Si la mediana de X es menor, están en el campo izquierdo (X < 0.5) y atacan hacia la derecha (+1)
    if home_median_x < away_median_x:
        attack_dir = {HOME_TAG: 1, AWAY_TAG: -1}
    else:
        attack_dir = {HOME_TAG: -1, AWAY_TAG: 1}

  
    # 2. Definir quien empieza con la posesión (jugador mas cercano a 0.5, 0.5)
    player_distances = []
    for pid in HOME_IDS + AWAY_IDS:
        px = row.get(f"{pid}_x", np.nan)
        py = row.get(f"{pid}_y", np.nan)
        if pd.notna(px) and pd.notna(py):
            dist = np.hypot(px - PITCH_CENTER_X, py - PITCH_CENTER_Y)
            team = HOME_TAG if pid in HOME_IDS else AWAY_TAG
            player_distances.append((dist, team, pid))
    # Ordenar por distancia al centro y coger el más cercano
    player_distances.sort(key=lambda x: x[0])
    starting_team = [item[1] for item in player_distances[:1]]

    # 3. Detectar cuando empieza a jugarse el balón (Cuando la pelota coge velocidad)
    window = df[
        (df["frame_id"] >= start_frame) & 
        (df["frame_id"] <= start_frame + max_wait_frames)
    ]
    dist_ball_center = np.hypot(
        window["ball_x"] - PITCH_CENTER_X, 
        window["ball_y"] - PITCH_CENTER_Y
    )
    # La pelota se juega cuando supera el umbral de velocidad 
    moving_mask = (window["ball_speed"] > TH_BALL_SPEED) 
    moving_frames = window.loc[moving_mask, "frame_id"]

    if not moving_frames.empty:
        play_start_frame = int(moving_frames.iloc[0])
    else:
        play_start_frame = start_frame

    return attack_dir, starting_team, play_start_frame

# ============================================
# PASO 2: ESTUDIAR Y DEFINIR JUEGO PROGRESIVO
# ============================================


# SANITY-CHECK PARAMETERS
# ==========================================
BOX_DEPTH = 16.5                 # penalty area depth (m)
BOX_HALF_WIDTH = 20.16           # half of penalty area width (m)  -> 40.32 m wide
POSSESSION_CHANGE_SECONDS = 5 # ball must stay with the other team this long to flip possession
CLOSE_DIST = 1.0                 # metres: nearest player only "owns" the ball if within this distance


def _infer_attack_dir(df, start_frame, n=FPS):
    """Same idea as kick_off(): the team with the lower median x attacks to the right (+1)."""
    w = df[(df["frame_id"] >= start_frame) & (df["frame_id"] < start_frame + n)]
    hx = w.reindex(columns=[f"{i}_x" for i in HOME_IDS]).to_numpy(float)
    ax_ = w.reindex(columns=[f"{i}_x" for i in AWAY_IDS]).to_numpy(float)
    h, a = np.nanmedian(hx), np.nanmedian(ax_)
    return {HOME_TAG: 1, AWAY_TAG: -1} if h < a else {HOME_TAG: -1, AWAY_TAG: 1}


def _box_mask(px, py, side):
    """Boolean (n, players) mask: player inside the penalty area at the right (+1) or left (-1) end."""
    in_y = (py >= PITCH_WIDTH / 2 - BOX_HALF_WIDTH) & (py <= PITCH_WIDTH / 2 + BOX_HALF_WIDTH)
    in_x = px >= PITCH_LENGTH - BOX_DEPTH if side == 1 else px <= BOX_DEPTH
    return in_x & in_y          # NaN comparisons are False -> untracked players never count


def _track_possession(frame_ids, near_team, hold_frames, initial=-1, retroactive=True):
    """Possession state machine.

    near_team: 0 = home, 1 = away, -1 = unknown (ball missing or no player close enough).
    Possession flips only after the ball has been closest to the OTHER team, without interruption
    by the current team, for >= hold_frames (measured with frame_id, so missing-ball frames still
    count as elapsed time). Unknown frames neither extend nor break a run.
    With retroactive=True the new possession is back-dated to the start of that run.
    """
    n = len(frame_ids)
    poss = np.full(n, -1, dtype=int)
    change = np.zeros(n, dtype=bool)
    current, run_start = initial, None
    for k in range(n):
        t = near_team[k]
        if t != -1:
            if current == -1:
                current = t
            elif t == current:
                run_start = None
            else:
                if run_start is None:
                    run_start = k
                if frame_ids[k] - frame_ids[run_start] >= hold_frames:
                    start = run_start if retroactive else k
                    poss[start:k] = t
                    change[start] = True
                    current, run_start = t, None
        poss[k] = current
    return poss, change


def sanity_check(df, halves=None, initial_team=None, attack_dir=None,
                 hold_seconds=POSSESSION_CHANGE_SECONDS, close_dist=CLOSE_DIST,
                 retroactive=True):
    """Frame-by-frame description of what is happening on the pitch (in-play frames only).

    Args:
        halves: {1: (start, end), 2: (start, end)} in frame_id. Defaults to real_halves.
        initial_team: {1: "H", 2: "A"} team that kicks off each half (str or list, e.g. from
                      kick_off()). None -> possession starts at the first close player.
        attack_dir: {1: {"H": 1, "A": -1}, 2: {...}} (+1 = attacks towards x=105). None -> inferred
                    from the player positions at each half's start frame.

    Returns a DataFrame with one row per frame:
        nearest_player / nearest_team / nearest_dist   closest player to the ball
        possession_team / possession_change            state machine output
        att_in_box / def_in_box                        possession team's players / opponents inside
                                                       the penalty area the possession team attacks
        home_in_att_box / away_in_att_box              each team's players in the box it attacks
        label_*                                        your dataframe's labels
        team_match / count_match                       label vs recomputed (<NA> if no label)
    """
    if halves is None:
        halves = real_halves
    ids = HOME_IDS + AWAY_IDS
    n_home = len(HOME_IDS)
    hold_frames = hold_seconds * FPS
    tags = np.array([HOME_TAG, AWAY_TAG, None], dtype=object)   # index -1 -> None
    parts = []

    for half, (start, end) in halves.items():
        sub = df[(df["frame_id"] >= start) & (df["frame_id"] <= end)].reset_index(drop=True)
        n = len(sub)
        if n == 0:
            continue

        def arr(cols):
            return sub.reindex(columns=cols).to_numpy(float)

        PX = arr([f"{i}_x" for i in ids]) * PITCH_LENGTH
        PY = arr([f"{i}_y" for i in ids]) * PITCH_WIDTH
        BX = arr(["ball_x"])[:, 0] * PITCH_LENGTH
        BY = arr(["ball_y"])[:, 0] * PITCH_WIDTH
        frame_ids = sub["frame_id"].to_numpy()

        # ---- 1. closest player to the ball
        d = np.hypot(PX - BX[:, None], PY - BY[:, None])      # NaN if player or ball missing
        d = np.where(np.isnan(d), np.inf, d)
        j = d.argmin(axis=1)
        dmin = d[np.arange(n), j]
        valid = np.isfinite(dmin)
        near_team = np.where(valid, np.where(j < n_home, 0, 1), -1)
        near_for_poss = np.where(valid & (dmin <= close_dist), near_team, -1)

        # ---- 2. possession with the 10 s rule
        init = initial_team.get(half) if initial_team else None
        if isinstance(init, (list, tuple)):
            init = init[0] if init else None
        initial = {HOME_TAG: 0, AWAY_TAG: 1}.get(init, -1)
        poss, change = _track_possession(frame_ids, near_for_poss, hold_frames, initial, retroactive)

        # ---- 3. players in the box
        dirs = (attack_dir or {}).get(half) or _infer_attack_dir(df, start)
        dh, da = dirs[HOME_TAG], dirs[AWAY_TAG]
        PXh, PYh, PXa, PYa = PX[:, :n_home], PY[:, :n_home], PX[:, n_home:], PY[:, n_home:]
        home_att = _box_mask(PXh, PYh, dh).sum(1)      # home players in the box home attacks
        home_def = _box_mask(PXh, PYh, -dh).sum(1)     # home players in their own box
        away_att = _box_mask(PXa, PYa, da).sum(1)
        away_def = _box_mask(PXa, PYa, -da).sum(1)
        att_in_box = np.where(poss == 0, home_att, np.where(poss == 1, away_att, np.nan))
        def_in_box = np.where(poss == 0, away_def, np.where(poss == 1, home_def, np.nan))

        # ---- 4. compare with the labels in the dataframe
        def col(name):
            return sub[name] if name in sub else pd.Series(np.nan, index=sub.index)

        lab_n = pd.to_numeric(col("players_in_the_box"), errors="coerce")
        lab_team = col("players_in_the_box_team").astype("string").str.strip()
        has_label = (lab_n.notna() & lab_team.notna()).to_numpy()
        lab_home = (lab_team == HOME_TAG).fillna(False).to_numpy()
        mine_for_label = np.where(lab_home, home_att, away_att)   # label team's players in the box it attacks
        count_match = pd.Series(pd.NA, index=sub.index, dtype="boolean")
        team_match = pd.Series(pd.NA, index=sub.index, dtype="boolean")
        count_match[has_label] = (mine_for_label == lab_n.to_numpy())[has_label]
        lab_team_arr = lab_team.fillna("").to_numpy(dtype=object)
        team_match[has_label] = (lab_team_arr == tags[poss])[has_label]

        parts.append(pd.DataFrame({
            "frame_id": frame_ids,
            "half": half,
            "timestamp": sub["timestamp"] if "timestamp" in sub else np.nan,
            "ball_tracked": ~(np.isnan(BX) | np.isnan(BY)),
            "nearest_player": pd.Series(np.where(valid, np.array(ids)[j], 0), dtype="Int64").where(valid),            "nearest_team": tags[np.where(valid, near_team, -1)],
            "nearest_dist": np.where(valid, dmin, np.nan),
            "possession_team": tags[poss],
            "possession_change": change,
            "n_tracked_home": np.isfinite(PXh).sum(1),
            "n_tracked_away": np.isfinite(PXa).sum(1),
            "home_in_att_box": home_att,
            "away_in_att_box": away_att,
            "att_in_box": att_in_box,
            "def_in_box": def_in_box,
            "set_piece": col("set_pieces").where(col("set_pieces").notna(), None),
            "label_players_in_box": lab_n,
            "label_box_team": lab_team,
            "team_match": team_match,
            "count_match": count_match,
        }))

    return pd.concat(parts, ignore_index=True)


def summarize_sanity(chk):
    """Quick health report of the output of sanity_check()."""
    print(f"In-play frames:            {len(chk):,}")
    print(f"Ball tracked:              {chk['ball_tracked'].mean():.1%}")
    print(f"Mean nearest-player dist:  {chk['nearest_dist'].mean():.2f} m")
    print(f"Possession changes:        {int(chk['possession_change'].sum())}")
    for half, g in chk.groupby("half"):
        print(f"   half {half}: {int(g['possession_change'].sum())} changes, "
              f"home {np.mean(g['possession_team'] == HOME_TAG):.1%} / "
              f"away {np.mean(g['possession_team'] == AWAY_TAG):.1%} of frames")
    lab = chk["team_match"].notna()
    if lab.any():
        print(f"Frames with a box label:   {lab.sum():,}")
        print(f"   label team == possession team: {chk.loc[lab, 'team_match'].astype(bool).mean():.1%}")
        print(f"   label count == recomputed:     {chk.loc[lab, 'count_match'].astype(bool).mean():.1%}")


def describe_frame(chk, frame_id):
    """One-line description of a frame, e.g. to print next to plot_frame()."""
    r = chk.loc[chk["frame_id"] == frame_id].iloc[0]
    return (f"frame {frame_id} | nearest: player {r['nearest_player']} ({r['nearest_team']}, "
            f"{r['nearest_dist']:.1f} m) | possession: {r['possession_team']} | "
            f"att/def in box: {r['att_in_box']}/{r['def_in_box']} | "
            f"label: {r['label_players_in_box']} ({r['label_box_team']}) | set piece: {r['set_piece']}")


df = pd.read_csv("data/dataframe_hackaton_final.csv",low_memory=False)
TEAM_MAP = {"Atletic Club": "A", "Barcelona": "H"}   # swap if the second one scored higher
df["players_in_the_box_team"] = df["players_in_the_box_team"].map(TEAM_MAP)
chk = sanity_check(df, halves=real_halves)
summarize_sanity(chk)

# What the raw labels look like, next to the ball-in-box labels
cols = ["frame_id", "ball_in_the_box", "ball_in_the_box_team",
        "players_in_the_box", "players_in_the_box_team"]
print(df.loc[df["players_in_the_box"].notna(), cols].head(20))

# Test hypotheses for the count
lab = chk[chk["label_players_in_box"].notna() & chk["possession_team"].notna()].copy()
n = pd.to_numeric(lab["label_players_in_box"])
print(n.value_counts().sort_index().head(15))
print(lab[["att_in_box", "def_in_box"]].describe())

hyp = {
    "label == attackers in box":      n == lab["att_in_box"],
    "label == defenders in box":      n == lab["def_in_box"],
    "label == attackers + defenders": n == lab["att_in_box"] + lab["def_in_box"],
    "label within +-1 of attackers":  (n - lab["att_in_box"]).abs() <= 1,
}
for k, v in hyp.items():
    print(f"{k}: {v.mean():.1%}")

# Test hypotheses for the team
t = lab["label_box_team"]
print("label team == possession:    ", f"{(t == lab['possession_team']).mean():.1%}")
print("label team == nearest team:  ", f"{(t == lab['nearest_team']).mean():.1%}")
# ==========================================
# PASO 3: DETECTAR LOS SET PIECES
# ==========================================




# ==========================================
# ==========================================
# ------- ALGORITMO FINAL -----------------
# ==========================================
# ==========================================