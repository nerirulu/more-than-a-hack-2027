import numpy as np

def get_players_and_ball(
    atd_lines,
    frame_number,
    team_map
):

    line = atd_lines[
        frame_number
    ].strip()

    parts = line.split(":")

    players_raw = parts[1].split(";")

    players = []

    for track_idx, entry in enumerate(players_raw):

        if entry.startswith("NaN"):
            continue

        try:

            x, y, speed = entry.split(",")

            track_id = track_idx + 1

            players.append(
                {
                    "id": track_id,
                    "team": team_map.get(track_id, -1),
                    "x": float(x),
                    "y": float(y),
                    "position": np.array(
                        [
                            float(x),
                            float(y)
                        ]
                    )
                }
            )

        except:

            pass

    ball = None

    try:

        bx, by, bs = parts[2].split(",")

        if bx != "NaN":

            ball = np.array(
                [
                    float(bx),
                    float(by)
                ]
            )

    except:

        pass

    return players, ball