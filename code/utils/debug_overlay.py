import cv2


def draw_debug_panel(
    frame,
    frame_idx,
    possession=None,
    ball_visible=None,
    ball_position=None,
    owner_team=None,
    owner_distance=None,
    smart_events=None
):

    panel_width = 520

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (panel_width, 350),
        (0, 0, 0),
        -1
    )

    alpha = 0.75

    frame[:] = cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1 - alpha,
        0
    )

    y = 30

    def line(text):

        nonlocal y

        cv2.putText(
            frame,
            str(text),
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,255),
            2
        )

        y += 25

    line(
        f"Frame: {frame_idx}"
    )

    y += 10

    line(
        f"Possession: {possession}"
    )

    line(
        f"Ball Visible: {ball_visible}"
    )

    if owner_team is not None:

        line(
            f"Nearest Team: {owner_team}"
        )

    if owner_distance is not None:

        line(
            f"Distance: {owner_distance:.4f}"
        )

    y += 10

    line("SMART TAGS")

    if smart_events:

        for event in smart_events[:6]:

            text = (
                f"{event['code']} | "
                f"{event['team']}"
            )

            line(text)

    return frame