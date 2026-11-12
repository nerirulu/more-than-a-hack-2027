from utils.player_utils import team_density_around_ball
import numpy as np

def ball_speed(
    current_ball,
    previous_ball,
    fps=25
):
    if current_ball is None:
        return None

    if previous_ball is None:
        return None

    distance = np.linalg.norm(
        current_ball -
        previous_ball
    )

    return distance * fps

def ball_outside_pitch(
    ball,
    margin=0.01
):

    if ball is None:
        return False

    x, y = ball

    return (
        x < -margin
        or x > 1 + margin
        or y < -margin
        or y > 1 + margin
    )

def classify_exit_side(
    ball
):

    x, y = ball

    if y < 0:
        return "top"

    if y > 1:
        return "bottom"

    if x < 0:
        return "left_goal"

    if x > 1:
        return "right_goal"

    return None

def contested_ball(
    ball,
    players,
    radius=0.05
):

    home, away = team_density_around_ball(
        ball,
        players,
        radius
    )

    return (
        home >= 2
        and
        away >= 2
    )

def near_sideline(
    ball,
    threshold=0.03
):
    """
    Ball close to top or bottom sideline.
    """

    if ball is None:
        return False

    x, y = ball

    return (
        y < threshold
        or
        y > 1 - threshold
    )

def near_corner(
    ball,
    threshold=0.05
):
    """
    Ball close to any corner.

    Returns
    -------
    bool
    """

    if ball is None:
        return False

    x, y = ball

    return (
        (x < threshold and y < threshold)
        or
        (x < threshold and y > 1 - threshold)
        or
        (x > 1 - threshold and y < threshold)
        or
        (x > 1 - threshold and y > 1 - threshold)
    )

def corner_id(
    ball,
    threshold=0.05
):
    """
    Returns corner name.
    """

    if ball is None:
        return None

    x, y = ball

    if x < threshold and y < threshold:
        return "top_left"

    if x < threshold and y > 1 - threshold:
        return "bottom_left"

    if x > 1 - threshold and y < threshold:
        return "top_right"

    if x > 1 - threshold and y > 1 - threshold:
        return "bottom_right"

    return None

def near_goal_line(
    ball,
    threshold=0.03
):
    """
    Close to x=0 or x=1.
    """

    if ball is None:
        return False

    x, y = ball

    return (
        x < threshold
        or
        x > 1 - threshold
    )

def ball_velocity(
    current_ball,
    previous_ball
):
    """
    Returns velocity vector.
    """

    if current_ball is None:
        return None

    if previous_ball is None:
        return None

    return current_ball - previous_ball

def moving_towards_goal_line(
    previous_ball,
    current_ball
):
    """
    Returns:
        left_goal
        right_goal
        None
    """

    if previous_ball is None:
        return None

    if current_ball is None:
        return None

    dx = (
        current_ball[0]
        - previous_ball[0]
    )

    if dx < 0:
        return "left_goal"

    if dx > 0:
        return "right_goal"

    return None