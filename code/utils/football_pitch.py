import matplotlib.pyplot as plt
from matplotlib.patches import (
    Rectangle,
    Circle,
    Arc
)


TEAM_COLORS = {
    0: "#1f77ff",   # blue
    1: "#ff3333",   # red
    -1: "#ffff00"   # unknown
}


def draw_pitch(
    ax=None,
    field_color="#2e8b57"
):
    """
    Draw football pitch
    coordinates:
        x -> [0,1]
        y -> [0,1]
    """

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(12,8)
        )

    ax.set_facecolor(
        field_color
    )

    # outer boundary

    ax.add_patch(
        Rectangle(
            (0,0),
            1,
            1,
            fill=False,
            ec="white",
            lw=2
        )
    )

    # halfway line

    ax.plot(
        [0.5,0.5],
        [0,1],
        color="white",
        lw=2
    )

    # centre circle

    ax.add_patch(
        Circle(
            (0.5,0.5),
            0.09,
            fill=False,
            ec="white",
            lw=2
        )
    )

    ax.add_patch(
        Circle(
            (0.5,0.5),
            0.005,
            color="white"
        )
    )

    # left penalty box

    ax.add_patch(
        Rectangle(
            (0.0,0.21),
            0.18,
            0.58,
            fill=False,
            ec="white",
            lw=2
        )
    )

    # right penalty box

    ax.add_patch(
        Rectangle(
            (0.82,0.21),
            0.18,
            0.58,
            fill=False,
            ec="white",
            lw=2
        )
    )

    # left 6-yard

    ax.add_patch(
        Rectangle(
            (0.0,0.36),
            0.06,
            0.28,
            fill=False,
            ec="white",
            lw=2
        )
    )

    # right 6-yard

    ax.add_patch(
        Rectangle(
            (0.94,0.36),
            0.06,
            0.28,
            fill=False,
            ec="white",
            lw=2
        )
    )

    # penalty spots

    ax.add_patch(
        Circle(
            (0.12,0.5),
            0.003,
            color="white"
        )
    )

    ax.add_patch(
        Circle(
            (0.88,0.5),
            0.003,
            color="white"
        )
    )

    # left D

    ax.add_patch(
        Arc(
            (0.12,0.5),
            0.22,
            0.22,
            angle=0,
            theta1=310,
            theta2=50,
            color="white",
            lw=2
        )
    )

    # right D

    ax.add_patch(
        Arc(
            (0.88,0.5),
            0.22,
            0.22,
            angle=0,
            theta1=130,
            theta2=230,
            color="white",
            lw=2
        )
    )

    ax.set_xlim(
        -0.02,
        1.02
    )

    ax.set_ylim(
        -0.02,
        1.02
    )

    ax.set_aspect(
        "equal"
    )

    ax.axis("off")

    return ax

def plot_frame_state(
    players,
    ball=None,
    ax=None
):

    ax = draw_pitch(ax)

    for player in players:

        x = player["x"]
        y = player["y"]

        team = player.get(
            "team",
            -1
        )

        color = TEAM_COLORS.get(
            team,
            "#ffff00"
        )

        ax.scatter(
            x,
            y,
            s=250,
            c=color,
            edgecolors="black",
            linewidths=1.5,
            zorder=10
        )

    if ball is not None:

        bx, by = ball

        ax.scatter(
            bx,
            by,
            s=120,
            c="white",
            edgecolors="black",
            linewidths=2,
            zorder=20
        )

    return ax

import cv2
import numpy as np


PITCH_WIDTH = 700
PITCH_HEIGHT = 450


def render_pitch_cv(
    players,
    ball=None,
    width=PITCH_WIDTH,
    height=PITCH_HEIGHT
):

    pitch = np.zeros(
        (
            height,
            width,
            3
        ),
        dtype=np.uint8
    )

    pitch[:] = (
        46,
        139,
        87
    )

    white = (
        255,
        255,
        255
    )

    # Outer lines

    cv2.rectangle(
        pitch,
        (10, 10),
        (width - 10, height - 10),
        white,
        2
    )

    # Midline

    cv2.line(
        pitch,
        (width // 2, 10),
        (width // 2, height - 10),
        white,
        2
    )

    # Centre circle

    cv2.circle(
        pitch,
        (width // 2, height // 2),
        int(height * 0.12),
        white,
        2
    )

    # Left penalty

    cv2.rectangle(
        pitch,
        (10, int(height * 0.21)),
        (int(width * 0.18), int(height * 0.79)),
        white,
        2
    )

    # Right penalty

    cv2.rectangle(
        pitch,
        (int(width * 0.82), int(height * 0.21)),
        (width - 10, int(height * 0.79)),
        white,
        2
    )

    # Draw players

    for player in players:

        x = player["x"]

        y = player["y"]

        team = player.get(
            "team",
            -1
        )

        px = int(
            x * width
        )

        py = int(
            y * height
        )

        if team == 0:

            color = (
                255,
                0,
                0
            )

        elif team == 1:

            color = (
                0,
                0,
                255
            )

        else:

            color = (
                0,
                255,
                255
            )

        cv2.circle(
            pitch,
            (px, py),
            8,
            color,
            -1
        )

    if ball is not None:

        bx = int(ball[0] * width)

        by = int(ball[1] * height)

        cv2.circle(
            pitch,
            (bx, by),
            6,
            (255,255,255),
            -1
        )

        cv2.circle(
            pitch,
            (bx, by),
            8,
            (0,0,0),
            2
        )

    return pitch