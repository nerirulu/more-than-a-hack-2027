import numpy as np

def nearest_player(
    ball,
    players
):

    best_idx = None

    best_distance = np.inf

    for idx, player in enumerate(players):

        distance = np.linalg.norm(
            player["position"] -
            ball
        )

        if distance < best_distance:

            best_distance = distance

            best_idx = idx

    return best_idx, best_distance

def nearest_team(
    ball,
    players
):

    idx, distance = nearest_player(
        ball,
        players
    )

    if idx is None:
        return None

    return players[idx]["team"]

def players_within_radius(
    centre,
    players,
    radius
):

    count = 0

    for player in players:

        d = np.linalg.norm(
            player["position"]
            -
            centre
        )

        if d < radius:
            count += 1

    return count

def team_density_around_ball(
    ball,
    players,
    radius=0.05
):

    home = 0
    away = 0

    for player in players:

        d = np.linalg.norm(
            player["position"]
            -
            ball
        )

        if d > radius:
            continue

        if player["team"] == 0:
            home += 1

        elif player["team"] == 1:
            away += 1

    return home, away

def mean_player_speed(
    current_players,
    previous_players,
    fps=25
):

    speeds = []

    for p1, p0 in zip(
        current_players,
        previous_players
    ):

        speed = np.linalg.norm(
            p1["position"]
            - p0["position"]
        ) * fps

        speeds.append(speed)

    if len(speeds) == 0:
        return None

    return np.mean(speeds)

def deepest_player(
    players,
    team
):
    """
    Return player closest to own goal.
    """

    candidates = [
        p
        for p in players
        if p["team"] == team
    ]

    if len(candidates) == 0:
        return None

    xs = np.array(
        [
            p["position"][0]
            for p in candidates
        ]
    )

    # team 0 defending left

    if team == 0:

        idx = np.argmin(xs)

    else:

        idx = np.argmax(xs)

    return candidates[idx]