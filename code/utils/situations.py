import numpy as np

from utils.ball_utils import (
    corner_id,
    near_sideline,
    ball_speed
)


THROW_IN_MIN_DEAD_FRAMES = 25
THROW_IN_RESTART_SPEED = 0.03

def detect_throw_in(
    frames
):
    """
    Parameters
    ----------
    frames : list

    Each element:

    {
        "frame": frame_id,
        "ball": np.array([x,y]) or None,
        "ball_speed": float or None
    }

    Returns
    -------
    list

    [
        {
            "start_frame": ...,
            "end_frame": ...
        }
    ]
    """

    throw_ins = []

    in_dead_ball = False

    dead_start = None

    for i in range(1, len(frames)):

        current = frames[i]

        previous = frames[i - 1]

        frame_id = current["frame"]

        ball = current["ball"]

        prev_ball = previous["ball"]

        speed = current["ball_speed"]

        # ====================================
        # START THROW-IN
        # ====================================

        if not in_dead_ball:

            exited_sideline = (
                prev_ball is not None
                and
                ball is not None
                and
                near_sideline(prev_ball)
                and
                (
                    ball[1] < 0
                    or
                    ball[1] > 1
                )
            )

            disappeared_near_sideline = (
                prev_ball is not None
                and
                ball is None
                and
                near_sideline(prev_ball)
            )

            if (
                exited_sideline
                or
                disappeared_near_sideline
            ):

                dead_start = frame_id

                in_dead_ball = True

                continue

        # ====================================
        # END THROW-IN
        # ====================================

        else:

            dead_duration = (
                frame_id
                -
                dead_start
            )

            if (
                dead_duration
                >= THROW_IN_MIN_DEAD_FRAMES
                and
                speed is not None
                and
                speed > THROW_IN_RESTART_SPEED
            ):

                throw_ins.append(
                    {
                        "start_frame": dead_start,
                        "end_frame": frame_id
                    }
                )

                in_dead_ball = False

                dead_start = None

    return throw_ins

from utils.ball_utils import (
    near_corner,
    near_goal_line
)


CORNER_MIN_DEAD_FRAMES = 25

CORNER_RESTART_SPEED = 0.03

def detect_corner(
    frames
):
    """
    frames:

    [
        {
            "frame": ...,
            "ball": ...,
            "ball_speed": ...
        }
    ]
    """

    corners = []

    in_corner = False

    dead_start = None

    corner_location = None

    for i in range(1, len(frames)):

        current = frames[i]

        previous = frames[i - 1]

        ball = current["ball"]

        prev_ball = previous["ball"]

        speed = current["ball_speed"]

        frame_id = current["frame"]

        # =====================================
        # START
        # =====================================

        if not in_corner:

            disappeared_near_corner = (

                prev_ball is not None

                and

                near_corner(prev_ball)

                and

                ball is None
            )

            exited_goal_line = (

                prev_ball is not None

                and

                near_goal_line(prev_ball)

                and

                ball is not None

                and

                (
                    ball[0] < 0
                    or
                    ball[0] > 1
                )
            )

            if (
                disappeared_near_corner
                or
                exited_goal_line
            ):

                in_corner = True

                dead_start = frame_id

                corner_location = prev_ball

                continue

        # =====================================
        # END
        # =====================================

        else:

            dead_duration = (
                frame_id
                -
                dead_start
            )

            if (
                dead_duration >= CORNER_MIN_DEAD_FRAMES
                and
                ball is not None
                and
                near_corner(ball)
                and
                speed is not None
                and
                speed > CORNER_RESTART_SPEED
            ):

                corners.append(
                    {
                        "start_frame": dead_start,
                        "end_frame": frame_id,
                        "corner": corner_id(
                            ball
                        )
                    }
                )

                in_corner = False

                dead_start = None

                corner_location = None

    return corners

from utils.ball_utils import (
    near_goal_line,
    moving_towards_goal_line
)

from utils.player_utils import (
    nearest_player
)

GOAL_KICK_MIN_DEAD_FRAMES = 25

GOAL_KICK_RESTART_SPEED = 0.03

DEFENDER_RADIUS = 0.08

def detect_goal_kick(
    frames
):
    """
    Returns:

    [
        {
            "start_frame": ...,
            "end_frame": ...
        }
    ]
    """

    intervals = []

    in_goal_kick = False

    start_frame = None

    exit_side = None

    for i in range(1, len(frames)):

        current = frames[i]

        previous = frames[i - 1]

        frame_id = current["frame"]

        ball = current["ball"]

        prev_ball = previous["ball"]

        speed = current["ball_speed"]

        # =====================================
        # START
        # =====================================

        if not in_goal_kick:

            exited_goal_line = (

                prev_ball is not None

                and

                near_goal_line(prev_ball)

                and

                ball is not None

                and

                (
                    ball[0] < 0
                    or
                    ball[0] > 1
                )
            )

            disappeared_to_goal_line = (

                prev_ball is not None

                and

                ball is None

                and

                near_goal_line(prev_ball)
            )

            if (
                exited_goal_line
                or
                disappeared_to_goal_line
            ):

                start_frame = frame_id

                in_goal_kick = True

                if prev_ball[0] < 0.5:
                    exit_side = "left"

                else:
                    exit_side = "right"

                continue

        # =====================================
        # END
        # =====================================

        else:

            dead_duration = (
                frame_id
                -
                start_frame
            )

            if (
                dead_duration
                <
                GOAL_KICK_MIN_DEAD_FRAMES
            ):
                continue

            if ball is None:
                continue

            if speed is None:
                continue

            if speed < GOAL_KICK_RESTART_SPEED:
                continue

            intervals.append(
                {
                    "start_frame": start_frame,
                    "end_frame": frame_id,
                    "exit_side": exit_side,
                    "type": "goal_kick"
                }
            )

            in_goal_kick = False

            start_frame = None

            exit_side = None

    return intervals