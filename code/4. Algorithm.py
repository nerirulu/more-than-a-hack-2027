import numpy as np
import pandas as pd
from tqdm import tqdm

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
SANITY_RATIO = 0.6        # 60% de la ventana
MARGIN_END_FRAMES = 25*15
BALL_MISSING_DEADBALL_FRAMES = 25 * 10

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

X_MIN, X_MAX = 0.00, 1.0  # Umbrales para considerar que sale por la línea de fondo
Y_MIN, Y_MAX = 0.00, 1.0 # Umbrales para considerar que sale por el lateral
LOOKBACK_FRAMES = 700      # 10 segundos a 25 FPS


# ==========================================
# PASO 0: PONER LAS 'D' Y SEPARAR PARTES
# ==========================================
def init_frames_with_X(halves_dict=real_halves, n_video_frames=VIDEO_FRAMES):
    """
    Inicializa el DataFrame de predicciones para todo el vídeo.
    Etiqueta con 'D' los tramos fuera de juego (pre-partido, descanso y post-partido)
    y deja en None los frames correspondientes a la 1ª y 2ª parte.

    Args:
        halves_dict (dict): Diccionario {1: (start_1, end_1), 2: (start_2, end_2)} en frames reales.
        n_video_frames (int): Número total de frames del vídeo.

    Returns:
        pd.DataFrame: DataFrame con columnas ['frame', 'label'].
    """

    # 1. Crear el DataFrame con las dos columnas: 'frame' (0 a n_video_frames - 1) y 'label'
    # Inicializamos todo con "D" por defecto
    df_results = pd.DataFrame({
        "frame": range(n_video_frames),
        "label": "D"
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
    starting_team = player_distances[0][1]

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

    pbar = tqdm(
        total=n,
        desc=f"Sanity {start_frame}-{end_frame}",
        unit="frames",
        dynamic_ncols=True
    )
    
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
        pbar.update(1)

    pbar.update(1)
    return df_results

# ==========================================
# PASO 3: DETECTAR LOS SET PIECES
# ==========================================

# SACAR LOS BOLEANOS MIRANDO LOS FRAMES PREVIOS

def extract_set_piece_booleans(df_part, current_idx):
    """
    Analiza los últimos 250 frames hasta el índice actual (current_idx) 
    adaptado a coordenadas normalizadas de 0 a 1.
    """
    start_win = max(0, current_idx - LOOKBACK_FRAMES)
    window_df = df_part.iloc[start_win : current_idx + 1]
    current_row = df_part.iloc[current_idx]
    
    # 1. PILOTA SURT PEL LATERAL (Y fuera de [0, 1])
    ball_out_lateral = False
    for _, row in window_df.iterrows():
        by = row.get("ball_y", np.nan)
        if pd.notna(by) and (by < 0.0 or by > 1.0):
            ball_out_lateral = True
            break

    # 2. PILOTA SURT PEL FONS (X fuera de [0, 1])
    ball_out_fondo = False
    for _, row in window_df.iterrows():
        bx = row.get("ball_x", np.nan)
        if pd.notna(bx) and (bx < 0.0 or bx > 1.0):
            ball_out_fondo = True
            break

    # 3. AGLOMERACIÓN CÓRNER (Jugadores en las 4 esquinas del campo 0-1)
    corner_frames_count = 0
    for _, row in window_df.iterrows():
        c_home, c_away = 0, 0
        for i in range(1, 25): # Home
            px, py = row.get(f"{i}_x", np.nan), row.get(f"{i}_y", np.nan)
            if pd.notna(px) and pd.notna(py) and (px < 0.1 or px > 0.9) and (py < 0.1 or py > 0.9):
                c_home += 1
        for i in range(25, 47): # Away
            px, py = row.get(f"{i}_x", np.nan), row.get(f"{i}_y", np.nan)
            if pd.notna(px) and pd.notna(py) and (px < 0.1 or px > 0.9) and (py < 0.1 or py > 0.9):
                c_away += 1
        if c_home >= 5 or c_away >= 5:
            corner_frames_count += 1
            
    aglomeracion_corner = (corner_frames_count >= 25)

    # 4. AGLOMERACIÓN MIG (Centro del campo: X entre 0.4 y 0.6)
    mig_frames_count = 0
    for _, row in window_df.iterrows():
        midfield_low_speed_count = 0
        for i in range(1, 47):
            px = row.get(f"{i}_x", np.nan)
            vx = row.get(f"{i}_x_vel", 0)
            vy = row.get(f"{i}_y_vel", 0)
            if pd.notna(px) and (0.4 <= px <= 0.6):
                speed = np.hypot(vx, vy) if pd.notna(vx) and pd.notna(vy) else 99
                if speed < 0.01:
                    midfield_low_speed_count += 1
        if midfield_low_speed_count >= 10:
            mig_frames_count += 1
            
    aglomeracion_mig = (mig_frames_count >= 25)

    # 5. JUGADOR AL FONDO
    lateral_frames_count = 0
    for _, row in window_df.iterrows():
        lateral_low_speed_count = 0
        for i in range(1, 47):
            px = row.get(f"{i}_x", np.nan)
            py = row.get(f"{i}_y", np.nan)
            vx = row.get(f"{i}_x_vel", 0)
            vy = row.get(f"{i}_y_vel", 0)
            if pd.notna(px) and pd.notna(py):
                speed = np.hypot(vx, vy) if pd.notna(vx) and pd.notna(vy) else 99
                if speed < 0.01 and (py <= 0.08 or py >= 0.92):
                    lateral_low_speed_count += 1
        if lateral_low_speed_count >= 7:
            lateral_frames_count += 1
    jugador_al_lateral = (lateral_frames_count >= 25)

    # 6. JUGADOR PARAT EN SEC (Velocidad casi 0 en escala 0-1)
    jugador_parat_en_sec = False
    
    # Umbrales de velocidad baja en escala normalizada (0 - 1)
    BALL_LOW_SPEED_THRESHOLD = 0.008
    PLAYER_LOW_SPEED_THRESHOLD = 0.015
    MIN_CONSECUTIVE_FRAMES = 50  # 2 segundos a 25 FPS
    
    consecutive_stopped_count = 0
    max_consecutive_stopped = 0
    
    for _, row in window_df.iterrows():
        bx = row.get("ball_x", np.nan)
        by = row.get("ball_y", np.nan)
        
        # 1. Obtener velocidad de la pelota
        b_speed = row.get("ball_speed", np.nan)
        if pd.isna(b_speed):
            bx_vel = row.get("ball_x_vel", 0)
            by_vel = row.get("ball_y_vel", 0)
            b_speed = np.hypot(bx_vel, by_vel) if pd.notna(bx_vel) and pd.notna(by_vel) else 99
            
        # 2. Encontrar el jugador más cercano a la pelota en este frame de la ventana
        closest_player_speed = 99
        if pd.notna(bx) and pd.notna(by):
            min_dist = float('inf')
            closest_idx = None
            
            for i in range(1, 47):
                px = row.get(f"{i}_x", np.nan)
                py = row.get(f"{i}_y", np.nan)
                if pd.notna(px) and pd.notna(py):
                    dist = np.hypot(px - bx, py - by)
                    if dist < min_dist:
                        min_dist = dist
                        closest_idx = i
                        
            # Si encontramos al jugador más cercano, calculamos su velocidad
            if closest_idx is not None:
                pvx = row.get(f"{closest_idx}_x_vel", 0)
                pvy = row.get(f"{closest_idx}_y_vel", 0)
                closest_player_speed = np.hypot(pvx, pvy) if pd.notna(pvx) and pd.notna(pvy) else 99
                
        # 3. Comprobar si ambos están parados en este frame
        if b_speed < BALL_LOW_SPEED_THRESHOLD and closest_player_speed < PLAYER_LOW_SPEED_THRESHOLD:
            consecutive_stopped_count += 1
        else:
            consecutive_stopped_count = 0  # Se rompe la racha si alguno acelera
            
        if consecutive_stopped_count > max_consecutive_stopped:
            max_consecutive_stopped = consecutive_stopped_count

    # Si la racha de parón conjunto ha durado más de 50 frames (2 segundos)
    if max_consecutive_stopped >= MIN_CONSECUTIVE_FRAMES:
        jugador_parat_en_sec = True

    # 7. Estimación por NaNs en la ventana (recorriendo cada frame)
    estimated_out_x = False
    estimated_out_y = False
    ball_loose = False
    
    # Iterar por la ventana buscando dónde aparecen los NaNs
    for idx_w in range(len(window_df)):
        row_w = window_df.iloc[idx_w]
        bx_w = row_w.get("ball_x", np.nan)
        by_w = row_w.get("ball_y", np.nan)
        
        # Si encontramos un NaN en la pelota dentro de la ventana
        if pd.isna(bx_w) or pd.isna(by_w):
            ball_loose = True  
            # Mirar el frame inmediatamente anterior dentro de la ventana para coger la última posición y velocidad válidas
            if idx_w > 0:
                prev_row = window_df.iloc[idx_w - 1]
                last_x = prev_row.get("ball_x", np.nan)
                last_y = prev_row.get("ball_y", np.nan)
                vx = prev_row.get("ball_x_vel", 0)
                vy = prev_row.get("ball_y_vel", 0)
                
                if pd.notna(last_x) and pd.notna(last_y):
                    # Si estaba cerca de los bordes y la velocidad lo sacaba:
                    if last_x < 0.15 or last_x > 0.85:
                        if pd.notna(vx) and vx != 0:
                            pred_x = last_x + (vx * (10))
                            if pred_x <= 0.0 or pred_x >= 1.0:
                                estimated_out_x = True
                                break
                                
                    if last_y < 0.15 or last_y > 0.85:
                        if pd.notna(vy) and vy != 0:
                            pred_y = last_y + (vy * (10))
                            if pred_y <= 0.0 or pred_y >= 1.0:
                                estimated_out_y = True
                                break


    return {
        "ball_out_lateral": ball_out_lateral,
        "ball_out_fondo": ball_out_fondo,
        "aglomeracion_corner": aglomeracion_corner,
        "aglomeracion_mig": aglomeracion_mig,
        "jugador_al_lateral": jugador_al_lateral,
        "jugador_parat_en_sec": jugador_parat_en_sec,
        "estimated_out_x": estimated_out_x,
        "estimated_out_y": estimated_out_y,
        "ball_loose": ball_loose}


# DETERMINAR EL SET PICE

def classify_set_piece_event(booleans_dict):
    """
    Determina el tipo de set piece basándose en el diccionario completo de booleanos,
    incluyendo la detección de Free Kick.
    """
    ball_out_lat = booleans_dict.get("ball_out_lateral", False)
    est_out_y = booleans_dict.get("estimated_out_y", False)
    ball_out_fon = booleans_dict.get("ball_out_fondo", False)
    est_out_x = booleans_dict.get("estimated_out_x", False)
    aglom_corner = booleans_dict.get("aglomeracion_corner", False)
    aglom_mig = booleans_dict.get("aglomeracion_mig", False)
    jugador_lat = booleans_dict.get("jugador_al_lateral", False)
    jugador_parado = booleans_dict.get("jugador_parat_en_sec", False)
    ball_loose = booleans_dict.get("ball_loose", False)
    
    # 1. THROW IN (Saque de banda)
    if ball_out_lat or (est_out_y and ball_loose):
        return "Throw In"
        
    # 2. CORNER / GOAL KICK (Saque de esquina o de puerta)
    if ball_out_fon or (est_out_x and ball_loose) and not aglom_corner:
        return "Corner / Goal Kick"
        
    # 3. KICK OFF AFTER GOAL (Saque de centro tras gol)
    if aglom_corner and (est_out_x and ball_loose):
        return "Kick Off After Goal"
        
    # 4. FREE KICK (Falta)
    if jugador_parado and not aglom_mig:
        return "Free Kick"
        
    # 5. PARADA DE JUEGO / BALÓN SUELTO (Interrupción general)
    if jugador_lat or aglom_mig:
        return "Parada de Juego"
        
    # 6. OPEN PLAY (Juego en curso por defecto)
    return None

# TROBAR ULTIM FRAME AMB PILOTA

def find_last_valid_ball_frame(
    df_part,
    current_idx
):

    idx = current_idx

    while idx >= 0:

        row = df_part.iloc[idx]

        bx = row.get("ball_x", np.nan)
        by = row.get("ball_y", np.nan)

        if pd.notna(bx) and pd.notna(by):

            return idx

        idx -= 1

    return 0

# TROBAR INICI EVENT

def find_event_onset_frame(df_part, current_idx, event_type):
    """
    Busca hacia atrás dentro de la ventana de los 250 frames el frame exacto 
    donde se originó el evento basándose en su disparatador físico.
    """

    LOOKBACK_FRAMES = 1000  # Por defecto

    if event_type ==  "Throw In":
        LOOKBACK_FRAMES = 400
    
    start_win = max(
        0,
        current_idx - LOOKBACK_FRAMES
    )

    # --------------------------------------------------
    # If current frame has no ball signal,
    # extend the window until the last valid ball signal
    # --------------------------------------------------

    current_row = df_part.iloc[current_idx]

    current_ball_visible = (
        pd.notna(current_row.get("ball_x", np.nan))
        and
        pd.notna(current_row.get("ball_y", np.nan))
    )

    if not current_ball_visible:

        last_ball_idx = find_last_valid_ball_frame(
            df_part,
            current_idx
        )

        start_win = min(
            start_win,
            max(
                0,
                last_ball_idx - 25
            )
        )

    window_df = df_part.iloc[
        start_win :
        current_idx + 1
    ]
    onset_idx = current_idx  # Por defecto el actual si no se halla el flanco exacto
    
    # 1. EVENTOS DE SALIDA DEL CAMPO (Throw In, Corner, Goal Kick, Kick Off)
    if event_type in ["Throw In", "Corner / Goal Kick", "Kick Off After Goal", "Goal Kick", "Corner Kick"]:
        # Buscamos el primer frame de la ventana donde la pelota cruza físicamente los límites
        for idx, row in window_df.iterrows():
            bx = row.get("ball_x", np.nan)
            by = row.get("ball_y", np.nan)
            # Si sale por bandas (Y) o fondos (X)
            if (pd.notna(bx) and (bx < 0.0 or bx > 1.0)) or (pd.notna(by) and (by < 0.0 or by > 1.0)):
                onset_idx = df_part.index.get_loc(idx)
                break
                
        # Si no salió físicamente por culpa de un NaN, buscamos dónde empezó la pérdida de señal / estimación
        if onset_idx == current_idx:
            for idx, row in window_df.iterrows():
                if pd.isna(row.get("ball_x")) or pd.isna(row.get("ball_y")):
                    onset_idx = df_part.index.get_loc(idx)
                    break

    # 2. FREE KICK (Falta) -> El jugador más cercano se para en seco
    elif event_type == "Free Kick":
        # Buscamos el inicio de la racha de frenazo del jugador más cercano al balón
        consecutive_stops = 0
        for idx, row in window_df.iterrows():
            bx, by = row.get("ball_x", np.nan), row.get("ball_y", np.nan)
            if pd.notna(bx) and pd.notna(by):
                # Encontrar distancia mínima al jugador
                min_dist = float('inf')
                c_idx = None
                for i in range(1, 47):
                    px, py = row.get(f"{i}_x", np.nan), row.get(f"{i}_y", np.nan)
                    if pd.notna(px) and pd.notna(py):
                        dist = np.hypot(px - bx, py - by)
                        if dist < min_dist:
                            min_dist = dist
                            c_idx = i
                            
                if c_idx is not None:
                    pvx = row.get(f"{c_idx}_x_vel", 0)
                    pvy = row.get(f"{c_idx}_y_vel", 0)
                    pspeed = np.hypot(pvx, pvy) if pd.notna(pvx) and pd.notna(pvy) else 99
                    
                    # Si el jugador baja drásticamente la velocidad (< 0.015)
                    if pspeed < 0.015:
                        consecutive_stops += 1
                        if consecutive_stops == 1: # ¡Primer frame del frenazo!
                            onset_idx = df_part.index.get_loc(idx)
                    else:
                        consecutive_stops = 0

    # 3. PARADA DE JUEGO -> Descenso general de velocidades
    elif event_type == "Parada de Juego":
        for idx, row in window_df.iterrows():
            b_speed = row.get("ball_speed", np.nan)
            if pd.isna(b_speed):
                bx_v = row.get("ball_x_vel", 0)
                by_v = row.get("ball_y_vel", 0)
                b_speed = np.hypot(bx_v, by_v) if pd.notna(bx_v) and pd.notna(by_v) else 99
                
            # Frame exacto donde la velocidad del balón cae por debajo del umbral de parón
            if b_speed < 0.008:
                onset_idx = df_part.index.get_loc(idx)
                break

    return onset_idx


# TROBAR FINAL I POSSESSIÓ 

def find_event_end_and_executor(df_part, current_idx, current_team):
    """
    Busca hacia adelante desde el índice actual el momento en que el balón 
    se pone en movimiento (fin del evento) e identifica qué jugador de qué equipo lo ejecuta.
    """
    # Miramos hacia adelante (ej. los siguientes 150 frames / 6 segundos)
    end_win = min(len(df_part), current_idx + 500)
    future_df = df_part.iloc[current_idx + 25 : end_win + 1]
    
    end_frame_id = None
    executor_player_id = None
    executor_team = None
    
    BALL_ACCEL_THRESHOLD = 0.1  # Umbral de velocidad para considerar que el balón ha salido disparado
    
    for idx, row in future_df.iterrows():
        bx = row.get("ball_x", np.nan)
        by = row.get("ball_y", np.nan)
        
        # 1. Obtener la velocidad del balón en este frame futuro
        b_speed = row.get("ball_speed", np.nan)
        if pd.isna(b_speed):
            bx_v = row.get("ball_x_vel", 0)
            by_v = row.get("ball_y_vel", 0)
            b_speed = np.hypot(bx_v, by_v) if pd.notna(bx_v) and pd.notna(by_v) else 0
            
        # 2. Comprobar si el balón se ha puesto en movimiento (supera el umbral de aceleración)
        if b_speed > BALL_ACCEL_THRESHOLD and pd.notna(bx) and pd.notna(by) and bx < 1.0 and bx > 0.0 and by > 0.0 and by < 1.0 :
            end_frame_id = row.get("frame_id", idx)
            
            # 3. Identificar quién pone el balón en movimiento (el jugador más cercano que se está moviendo)
            min_dist = float('inf')
            best_player = None
            
            for i in range(1, 47):
                px = row.get(f"{i}_x", np.nan)
                py = row.get(f"{i}_y", np.nan)
                if pd.notna(px) and pd.notna(py):
                    dist = np.hypot(px - bx, py - by)
                    if dist < min_dist:
                        min_dist = dist
                        best_player = i
                        
            if best_player is not None:
                executor_player_id = best_player
                # Asignar equipo según el ID del jugador (Home: 1-24, Away: 25-46)
                executor_team = "H" if executor_player_id <= 24 else "A"
                
            break # Encontramos el momento exacto del saque y rompemos el bucle
            
    # Fallback si no acelera claramente en la ventana futura
    if end_frame_id is None and len(future_df) > 0:
        end_frame_id = future_df.iloc[-1].get("frame_id", df_part.iloc[current_idx]["frame_id"])
        
    return {
        "end_frame": end_frame_id,
        "executor_player": executor_player_id,
        "executor_team": executor_team
    }

########################
# GOALS
########################

def detect_goal_event(row):
    """
    Uses Smart Tagging.

    Returns
    -------
    bool
    """

    goal_flag = pd.to_numeric(
        row.get("goals", 0),
        errors="coerce"
    )

    return goal_flag == 1

def find_goal_onset_frame(
    df_part,
    current_idx
):

    current_row = df_part.iloc[current_idx]

    bx = current_row.get(
        "ball_x",
        np.nan
    )

    by = current_row.get(
        "ball_y",
        np.nan
    )

    ball_visible = (
        pd.notna(bx)
        and
        pd.notna(by)
    )

    #
    # CASE 1
    # BALL VISIBLE
    #

    if ball_visible:

        start_win = max(
            0,
            current_idx - 3000
        )

    #
    # CASE 2
    # BALL LOST
    #

    else:

        last_ball_idx = (
            find_last_valid_ball_frame(
                df_part,
                current_idx
            )
        )

        start_win = max(
            0,
            last_ball_idx - 25
        )

    window_df = df_part.iloc[
        start_win:
        current_idx + 1
    ]

    onset_idx = current_idx

    for idx, row in window_df.iterrows():

        bx = row.get(
            "ball_x",
            np.nan
        )

        by = row.get(
            "ball_y",
            np.nan
        )

        if (
            pd.notna(bx)
            and
            (
                bx < 0.0
                or
                bx > 1.0
            )
        ):

            onset_idx = (
                df_part.index.get_loc(idx)
            )

            break

    if onset_idx == current_idx:

        for idx, row in window_df.iterrows():

            if (
                pd.isna(
                    row.get(
                        "ball_x"
                    )
                )
                or
                pd.isna(
                    row.get(
                        "ball_y"
                    )
                )
            ):

                onset_idx = (
                    df_part.index.get_loc(idx)
                )

                break

    return onset_idx

def find_goal_end_frame(
    df_part,
    current_idx
):

    idx = current_idx

    while idx < len(df_part):

        row = df_part.iloc[idx]

        bx = row.get(
            "ball_x",
            np.nan
        )

        by = row.get(
            "ball_y",
            np.nan
        )

        speed = row.get(
            "ball_speed",
            np.nan
        )

        if (
            pd.notna(bx)
            and
            pd.notna(by)
            and
            pd.notna(speed)
        ):

            if (
                speed > TH_BALL_SPEED
            ):

                return idx

        idx += 1

    return current_idx

def find_kickoff_team_after_goal(
    df_part,
    goal_end_idx
):

    row = df_part.iloc[goal_end_idx]

    bx = row["ball_x"]
    by = row["ball_y"]

    min_dist = float("inf")
    team = None

    for i in range(1, 47):

        px = row.get(f"{i}_x", np.nan)
        py = row.get(f"{i}_y", np.nan)

        if pd.notna(px) and pd.notna(py):

            dist = np.hypot(
                px - bx,
                py - by
            )

            if dist < min_dist:

                min_dist = dist

                team = (
                    HOME_TAG
                    if i <= 24
                    else AWAY_TAG
                )

    return team

# ==========================================
# ==========================================
# ------- ALGORITMO FINAL -----------------
# MAIN POSSESSION ENGINE
# ==========================================
# ==========================================

def build_possession_dataframe(df):

    print("Computing closest player per frame...")

    if "closest_team" not in df.columns:
        df = compute_closest_player_per_frame(df)

    df_results = init_frames_with_X()

    # =====================================
    # PASS 1: POSSESSION FOR BOTH HALVES
    # =====================================

    for half in [1, 2]:

        half_start, half_end = real_halves[half]

        print("\n" + "=" * 60)
        print(f"RUNNING SANITY CHECK HALF {half}")
        print("=" * 60)

        _, kickoff_team, _ = kick_off(
            df,
            half_start
        )

        df_results = run_final_sanity_check(
            df=df,
            df_results=df_results,
            start_frame=half_start,
            end_frame=half_end,
            initial_team=kickoff_team
        )

    print("\nInitial possession pass finished.")

    # =====================================
    # PASS 2: GOALS
    # =====================================

    print("\nProcessing goals...")

    goal_series = (
        pd.to_numeric(
            df["goals"],
            errors="coerce"
        )
        .fillna(0)
    )

    goal_rows = df.index[
        (goal_series == 1)
        &
        (
            goal_series.shift(
                1,
                fill_value=0
            ) == 0
        )
    ]

    print(
        f"Goal events detected: "
        f"{len(goal_rows)}"
    )

    processed_goal_frames = set()

    for idx in tqdm(
        goal_rows,
        desc="Goals"
    ):

        frame_id = int(
            df.iloc[idx]["frame_id"]
        )

        if frame_id in processed_goal_frames:
            continue

        half_found = None

        for half in [1, 2]:

            start_h, end_h = real_halves[half]

            if (
                frame_id >= start_h
                and
                frame_id <= end_h
            ):
                half_found = half
                break

        if half_found is None:
            continue

        start_h, end_h = real_halves[half_found]

        df_part = (
            df[
                df["frame_id"].between(
                    start_h,
                    end_h
                )
            ]
            .sort_values("frame_id")
            .reset_index(drop=True)
        )

        local_idx = np.abs(
            df_part["frame_id"].to_numpy()
            - frame_id
        ).argmin()

        goal_start_idx = find_goal_onset_frame(
            df_part,
            local_idx
        )

        goal_end_idx = find_goal_end_frame(
            df_part,
            local_idx
        )

        goal_start_frame = int(
            df_part.iloc[
                goal_start_idx
            ]["frame_id"]
        )

        goal_end_frame = int(
            df_part.iloc[
                goal_end_idx
            ]["frame_id"]
        )

        print(
            f"[GOAL] "
            f"{goal_start_frame} -> {goal_end_frame}"
        )

        df_results.loc[
            goal_start_frame:
            goal_end_frame,
            "label"
        ] = "D"

        processed_goal_frames.update(
            range(
                goal_start_frame,
                goal_end_frame + 1
            )
        )

    # =====================================
    # PASS 3: SET PIECES
    # =====================================

    print("\nProcessing set pieces...")

    set_piece_series = (
        df["set_pieces"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    setpiece_rows = df.index[
        (set_piece_series != "")
        &
        (
            set_piece_series
            !=
            set_piece_series.shift(
                1,
                fill_value=""
            )
        )
    ]

    print(
        f"Set piece events detected: "
        f"{len(setpiece_rows)}"
    )

    processed_setpiece_frames = set()

    for idx in tqdm(
        setpiece_rows,
        desc="Set Pieces"
    ):

        frame_id = int(
            df.iloc[idx]["frame_id"]
        )

        if frame_id in processed_setpiece_frames:
            continue

        half_found = None

        for half in [1, 2]:

            start_h, end_h = real_halves[half]

            if (
                frame_id >= start_h
                and
                frame_id <= end_h
            ):
                half_found = half
                break

        if half_found is None:
            continue

        start_h, end_h = real_halves[half_found]

        df_part = (
            df[
                df["frame_id"].between(
                    start_h,
                    end_h
                )
            ]
            .sort_values("frame_id")
            .reset_index(drop=True)
        )

        local_idx = np.abs(
            df_part["frame_id"].to_numpy()
            - frame_id
        ).argmin()

        booleans_dict = (
            extract_set_piece_booleans(
                df_part,
                local_idx
            )
        )

        event_type = (
            classify_set_piece_event(
                booleans_dict
            )
        )

        if event_type is None:
            continue

        event_start_idx = (
            find_event_onset_frame(
                df_part,
                local_idx,
                event_type
            )
        )

        event_info = (
            find_event_end_and_executor(
                df_part,
                local_idx,
                HOME_TAG
            )
        )

        event_end_frame = (
            event_info["end_frame"]
        )

        event_start_frame = int(
            df_part.iloc[
                event_start_idx
            ]["frame_id"]
        )

        print(
            f"[SET PIECE] "
            f"{event_type} "
            f"{event_start_frame} -> {event_end_frame}"
        )

        df_results.loc[
            event_start_frame:
            event_end_frame,
            "label"
        ] = "D"

        processed_setpiece_frames.update(
            range(
                event_start_frame,
                int(event_end_frame) + 1
            )
        )

    # =====================================
    # PASS 4: LONG BALL DISAPPEARANCES
    # =====================================

    print("\nProcessing long ball disappearances...")

    ball_visible = (
        df["ball_x"].notna()
        &
        df["ball_y"].notna()
    )

    start_missing = None

    for idx in tqdm(
        range(len(df)),
        desc="Ball Missing"
    ):

        is_missing = not ball_visible.iloc[idx]

        if is_missing:

            if start_missing is None:
                start_missing = idx

        else:

            if start_missing is not None:

                duration = idx - start_missing

                if duration >= BALL_MISSING_DEADBALL_FRAMES:

                    start_frame = int(
                        df.iloc[start_missing]["frame_id"]
                    )

                    end_frame = int(
                        df.iloc[idx - 1]["frame_id"]
                    )

                    print(
                        f"[BALL LOST] "
                        f"{start_frame} -> {end_frame} "
                        f"({duration} frames)"
                    )

                    df_results.loc[
                        start_frame:end_frame,
                        "label"
                    ] = "D"

                start_missing = None

    # Handle disappearance reaching end of dataframe

    if start_missing is not None:

        duration = len(df) - start_missing

        if duration >= BALL_MISSING_DEADBALL_FRAMES:

            start_frame = int(
                df.iloc[start_missing]["frame_id"]
            )

            end_frame = int(
                df.iloc[-1]["frame_id"]
            )

            print(
                f"[BALL LOST] "
                f"{start_frame} -> {end_frame} "
                f"({duration} frames)"
            )

            df_results.loc[
                start_frame:end_frame,
                "label"
            ] = "D"

    # =====================================
    # FINAL CLEANUP
    # =====================================

    df_results["label"] = (
        df_results["label"]
        .fillna("X")
    )

    print("\nFinished.")

    print(
        df_results["label"]
        .value_counts(dropna=False)
    )

    return df_results

#############################
# ------------- RUN -----------------
##############################

# Load your merged dataframe
df = pd.read_csv(
    "../data/dataframe_hackaton_final.csv"
)

df_submission = build_possession_dataframe(df)

print(df_submission.head())

print(
    df_submission["label"]
    .value_counts(dropna=False)
)

df_submission.to_csv(
    "submission.csv",
    index=False
)

print("Saved submission.csv")