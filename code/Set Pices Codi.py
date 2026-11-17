import numpy as np
import pandas as pd

# ==========================================
# CONFIGURACIÓN DE LÍMITES (NORMALIZADO 0 - 1)
# ==========================================
X_MIN, X_MAX = 0.00, 0.1  # Umbrales para considerar que sale por la línea de fondo
Y_MIN, Y_MAX = 0.00, 0.1 # Umbrales para considerar que sale por el lateral
LOOKBACK_FRAMES = 250      # 10 segundos a 25 FPS


# ==================================================
# SACAR LOS BOLEANOS MIRANDO LOS 250 FRAMES PREVIOS
# ==================================================

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
                    # Proyectar asumiendo 1 frame de salto (1/25 segundos) o usando las velocidades
                    # Si estaba cerca de los bordes y la velocidad lo sacaba:
                    if last_x < 0.15 or last_x > 0.85:
                        if pd.notna(vx) and vx != 0:
                            pred_x = last_x + (vx * (1.0 / 25.0))
                            if pred_x <= 0.0 or pred_x >= 1.0:
                                estimated_out_x = True
                                break
                                
                    if last_y < 0.15 or last_y > 0.85:
                        if pd.notna(vy) and vy != 0:
                            pred_y = last_y + (vy * (1.0 / 25.0))
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




# ==================================================
# DETERMINAR EL SET PICE
# ==================================================

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


# ==================================================
# TROBAR INICI EVENT
# ==================================================

def find_event_onset_frame(df_part, current_idx, event_type):
    """
    Busca hacia atrás dentro de la ventana de los 250 frames el frame exacto 
    donde se originó el evento basándose en su disparatador físico.
    """
    start_win = max(0, current_idx - 250)
    window_df = df_part.iloc[start_win : current_idx + 1]
    onset_idx = current_idx  # Por defecto el actual si no se halla el flanco exacto
    
    # 1. EVENTOS DE SALIDA DEL CAMPO (Throw In, Corner, Goal Kick, Kick Off)
    if event_type in ["Throw In", "Corner / Goal Kick", "Kick Off After Goal"]:
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


# ==================================================
# TROBAR FINAL I POSSESSIÓ 
# ==================================================
def find_event_end_and_executor(df_part, current_idx, current_team):
    """
    Busca hacia adelante desde el índice actual el momento en que el balón 
    se pone en movimiento (fin del evento) e identifica qué jugador de qué equipo lo ejecuta.
    """
    # Miramos hacia adelante (ej. los siguientes 150 frames / 6 segundos)
    end_win = min(len(df_part) - 1, current_idx + 150)
    future_df = df_part.iloc[current_idx : end_win + 1]
    
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
        if b_speed > BALL_ACCEL_THRESHOLD and pd.notna(bx) and 0 < bx < 1 and pd.notna(by) and 0 < by < 1:
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



