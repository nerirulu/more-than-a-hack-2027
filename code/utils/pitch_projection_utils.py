import cv2
import numpy as np


TEAM_COLORS = {
    0: (255, 0, 0),      # blue
    1: (0, 0, 255),      # red
    -1: (0, 255, 255)    # yellow
}


def project_point(
    H,
    x,
    y
):
    """
    Pitch coords [0,1]
    ->
    image coords [px]
    """

    denom = (
        H[2, 0] * x +
        H[2, 1] * y +
        H[2, 2]
    )

    px = (
        H[0, 0] * x +
        H[0, 1] * y +
        H[0, 2]
    ) / denom

    py = (
        H[1, 0] * x +
        H[1, 1] * y +
        H[1, 2]
    ) / denom

    return int(px), int(py)

def draw_players(
    frame,
    H,
    players,
    radius=10,
    show_ids=True
):
    """
    players:
    [
        {
            "id": 3,
            "x": 0.42,
            "y": 0.51,
            "team": 0
        }
    ]
    """

    img = frame.copy()

    for player in players:

        px, py = project_point(
            H,
            player["x"],
            player["y"]
        )

        team = player.get(
            "team",
            -1
        )

        color = TEAM_COLORS.get(
            team,
            TEAM_COLORS[-1]
        )

        cv2.circle(
            img,
            (px, py),
            radius,
            color,
            -1
        )

        cv2.circle(
            img,
            (px, py),
            radius,
            (0, 0, 0),
            2
        )

        if show_ids:

            cv2.putText(
                img,
                str(player["id"]),
                (px + 10, py),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

    return img

def draw_ball(
    frame,
    H,
    ball_position,
    color=(255,255,255)
):

    img = frame.copy()

    if ball_position is None:
        return img

    px, py = project_point(
        H,
        ball_position[0],
        ball_position[1]
    )

    cv2.circle(
        img,
        (px, py),
        12,
        color,
        -1
    )

    cv2.circle(
        img,
        (px, py),
        14,
        (0,0,0),
        2
    )

    return img

def draw_ball(
    frame,
    H,
    ball_position,
    color=(255,255,255)
):

    img = frame.copy()

    if ball_position is None:
        return img

    px, py = project_point(
        H,
        ball_position[0],
        ball_position[1]
    )

    cv2.circle(
        img,
        (px, py),
        12,
        color,
        -1
    )

    cv2.circle(
        img,
        (px, py),
        14,
        (0,0,0),
        2
    )

    return img

def draw_tracking_state(
    frame,
    H,
    players,
    ball_position=None,
    show_ids=True
):

    img = frame.copy()

    img = draw_players(
        img,
        H,
        players,
        show_ids=show_ids
    )

    img = draw_ball(
        img,
        H,
        ball_position
    )

    return img