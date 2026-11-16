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
SANITY_WINDOW_FRAMES = 75  # 3 segundos a 25 FPS
SANITY_RATIO = 0.75        # 75% de la ventana
MARGIN_END_FRAMES = 25*15

# Columnas tácticas del CSV
ATTACK_COLUMNS = [
    "progression", "attacking_trend", "build_up", 
    "long_ball", "creating_chances", "goals", 
    "ball_in_final_third", "ball_in_the_box", "players_in_the_box"
]

DEFENSE_COLUMNS = [
    "defending_in_middle_third", "defensive_transition", 
    "defending_in_attacking_third", "defending_in_defensive_third"
]




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


# FUNCIÓN: MAPEA EL EQUIPO CON LAS LABELS

def map_team_str_to_tag(val):
    """
    Traduce el string del dataset a etiqueta corta:
    - Si contiene 'ba'/'Ba' -> 'H'
    - Si contiene 'at'/'At' -> 'A'
    """
    if pd.isna(val):
        return None
    val_str = str(val)
    if "ba" in val_str.lower():
        return HOME_TAG
    elif "at" in val_str.lower():
        return AWAY_TAG
    return None


# FUNCIÓN: CALCULAR JUGADOR MÁS CERCANO POR FRAME

def compute_closest_player_per_frame(df):
    """
    Recorre todas las filas del DataFrame y calcula en cada frame 
    qué equipo tiene al jugador más cercano a la pelota.
    Devuelve la columna 'closest_team' con valores 'H' o 'A'.
    """
    n_rows = len(df)
    ball_x = df["ball_x"].to_numpy()
    ball_y = df["ball_y"].to_numpy()
    
    closest_team_arr = []
    
    # Pre-extraer coordenadas en arrays de numpy para que vaya rapidísimo
    home_x = [df.get(f"{i}_x", pd.Series([np.nan]*n_rows)).to_numpy() for i in range(1, 25)]
    home_y = [df.get(f"{i}_y", pd.Series([np.nan]*n_rows)).to_numpy() for i in range(1, 25)]
    
    away_x = [df.get(f"{i}_x", pd.Series([np.nan]*n_rows)).to_numpy() for i in range(25, 47)]
    away_y = [df.get(f"{i}_y", pd.Series([np.nan]*n_rows)).to_numpy() for i in range(25, 47)]
    
    for idx in range(n_rows):
        bx, by = ball_x[idx], ball_y[idx]
        
        # Si no hay coordenadas válidas de pelota en este frame, dejamos NaN
        if pd.isna(bx) or pd.isna(by):
            closest_team_arr.append(np.nan)
            continue
            
        min_dist = float('inf')
        closest_t = None
        
        # Buscar el jugador más cercano en el equipo Home (1-24)
        for hx_arr, hy_arr in zip(home_x, home_y):
            px, py = hx_arr[idx], hy_arr[idx]
            if not pd.isna(px) and not pd.isna(py):
                d = np.hypot(px - bx, py - by)
                if d < min_dist:
                    min_dist = d
                    closest_t = "H"
                    
        # Buscar el jugador más cercano en el equipo Away (25-46)
        for ax_arr, ay_arr in zip(away_x, away_y):
            px, py = ax_arr[idx], hy_arr[idx] if False else ax_arr[idx] # (seguridad de arrays)
            px, py = ax_arr[idx], ay_arr[idx]
            if not pd.isna(px) and not pd.isna(py):
                d = np.hypot(px - bx, py - by)
                if d < min_dist:
                    min_dist = d
                    closest_t = "A"
                    
        closest_team_arr.append(closest_t)
        
    df["closest_team"] = closest_team_arr
    return df


# FUNCIÓN PRINCIPAL: SANITY CHECK DE 3 NIVELES
def run_final_sanity_check(df, df_results, start_frame, end_frame, initial_team):
    """
    Jerarquía de 3 niveles con conversión robusta a numérico (evita errores de tipo object):
      1. Eventos tácticos del CSV (convirtiendo valores de object a numérico de forma segura).
      2. Jugador más cercano analizando los 75 frames anteriores.
      3. Fallback: Mantener el último equipo con posesión conocida.
    """
    current_team = initial_team
    
    if "closest_team" not in df.columns:
        df = compute_closest_player_per_frame(df)
        
    mask_part = df["frame_id"].between(start_frame, end_frame)
    df_part = df[mask_part].sort_values("frame_id")
    
    frames_list = df_part["frame_id"].to_numpy()
    closest_teams = df_part["closest_team"].to_numpy()
    row_dicts = df_part.to_dict("records")
    
    n = len(frames_list)
    i = 0
    while i < n:
        current_frame = int(frames_list[i])
        row_data = row_dicts[i]
        
        ball_is_valid = pd.notna(row_data.get("ball_x")) and pd.notna(row_data.get("ball_y"))
        
        # NIVEL 1: Eventos tácticos (con conversión segura a numérico)
        tactical_forced_team = None
        
        # Comprobar ataques
        for col in ATTACK_COLUMNS:
            val_raw = row_data.get(col, 0)
            # Convertir de object/string a numérico de forma segura
            val = pd.to_numeric(val_raw, errors='coerce')
            if pd.isna(val):
                val = 0
                
            # Regla especial para players_in_the_box: ignorar si es 1 o 2 (debe ser > 2)
            if col == "players_in_the_box":
                is_active = (val > 2)
            else:
                is_active = (val == 1)
                
            if is_active:
                # Look-ahead contando frames mientras siga activo
                remaining_frames = 0
                k = i
                while k < n:
                    k_val_raw = row_dicts[k].get(col, 0)
                    k_val = pd.to_numeric(k_val_raw, errors='coerce')
                    if pd.isna(k_val):
                        k_val = 0
                        
                    if col == "players_in_the_box":
                        if k_val <= 2:
                            break
                    else:
                        if k_val != 1:
                            break
                    remaining_frames += 1
                    k += 1
                
                # Si estamos a más de 15 segundos (375 frames) del final del evento, lo obedecemos
                if remaining_frames > MARGIN_END_FRAMES:
                    team_col = f"{col}_team"
                    if team_col in row_data:
                        mapped = map_team_str_to_tag(row_data[team_col])
                        if mapped is not None:
                            tactical_forced_team = mapped
                            break
                    
        # Comprobar defensas
        if tactical_forced_team is None:
            for col in DEFENSE_COLUMNS:
                val_raw = row_data.get(col, 0)
                val = pd.to_numeric(val_raw, errors='coerce')
                if pd.isna(val):
                    val = 0
                    
                if val == 1:
                    remaining_frames = 0
                    k = i
                    while k < n:
                        k_val_raw = row_dicts[k].get(col, 0)
                        k_val = pd.to_numeric(k_val_raw, errors='coerce')
                        if pd.isna(k_val):
                            k_val = 0
                        if k_val != 1:
                            break
                        remaining_frames += 1
                        k += 1
                    
                    if remaining_frames > MARGIN_END_FRAMES:
                        team_col = f"{col}_team"
                        if team_col in row_data:
                            def_tag = map_team_str_to_tag(row_data[team_col])
                            if def_tag is not None:
                                tactical_forced_team = AWAY_TAG if def_tag == HOME_TAG else HOME_TAG
                                break
                        
        if tactical_forced_team is not None:
            current_team = tactical_forced_team
            
        else:
            # NIVEL 2: Ventana de los 75 frames ANTERIORES (Geometría)
            if ball_is_valid and i >= SANITY_WINDOW_FRAMES:
                window_teams = closest_teams[i - SANITY_WINDOW_FRAMES : i]
                window_teams_clean = window_teams[pd.notna(window_teams)]
                
                if len(window_teams_clean) > 0:
                    rival_team = AWAY_TAG if current_team == HOME_TAG else HOME_TAG
                    rival_ratio = np.mean(window_teams_clean == rival_team)
                    
                    if rival_ratio >= SANITY_RATIO:
                        current_team = rival_team
                        
            # NIVEL 3: Fallback (Mantiene el `current_team` anterior)
            pass
        
        # Asignar la posesión resuelta al frame actual
        df_results.loc[current_frame, "label"] = current_team
        i += 1
        
    return df_results



# ==========================================
# PASO 3: DETECTAR LOS SET PIECES
# ==========================================




# ==========================================
# ==========================================
# ------- ALGORITMO FINAL -----------------
# ==========================================
# ==========================================