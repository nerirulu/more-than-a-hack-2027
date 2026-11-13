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




# ==========================================
# PASO 3: DETECTAR LOS SET PIECES
# ==========================================




# ==========================================
# ==========================================
# ------- ALGORITMO FINAL -----------------
# ==========================================
# ==========================================