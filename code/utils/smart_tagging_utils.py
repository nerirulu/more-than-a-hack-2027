from kloppy import sportscode
import pandas as pd


FPS = 25


def load_smart_tagging(xml_path):

    events = sportscode.load(
        xml_path
    ).to_df()

    events["start_frame"] = (
        events["timestamp"]
        .dt.total_seconds()
        .mul(FPS)
        .round()
        .astype(int)
    )

    events["end_frame"] = (
        events["end_timestamp"]
        .dt.total_seconds()
        .mul(FPS)
        .round()
        .astype(int)
    )

    return events


def build_frame_lookup(events):

    lookup = {}

    for idx, row in events.iterrows():

        start_frame = row["start_frame"]
        end_frame = row["end_frame"]

        event = {
            "code": row.get("code"),
            "team": row.get("Team"),
            "half": row.get("Half"),
            "type": row.get("Type"),
            "side": row.get("Side")
        }

        for frame in range(
            start_frame,
            end_frame + 1
        ):

            lookup.setdefault(
                frame,
                []
            ).append(event)

    return lookup


def get_active_events(
    frame_number,
    frame_lookup
):

    return frame_lookup.get(
        frame_number,
        []
    )

def build_frame_lookup(events):

    lookup = {}

    for _, row in events.iterrows():

        start_frame = row["start_frame"]
        end_frame = row["end_frame"]

        event = row.to_dict()

        for frame in range(
            start_frame,
            end_frame + 1
        ):

            lookup.setdefault(
                frame,
                []
            ).append(event)

    return lookup

def get_event_codes(
    frame_idx,
    frame_lookup
):

    return [
        e["code"]
        for e in frame_lookup.get(
            frame_idx,
            []
        )
    ]

