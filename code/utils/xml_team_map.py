import xml.etree.ElementTree as ET


BARCA_TEAM = "MET_user_team_o4Hossvb8M"
ATH_TEAM = "MET_user_team_x7C4FFX2HF"


def load_track_team_map(xml_path):

    tree = ET.parse(xml_path)

    root = tree.getroot()

    team_map = {}

    for player in root.findall(".//Player"):

        track_id = int(
            player.attrib["id"]
        )

        team_id = player.attrib["teamId"]

        if team_id == BARCA_TEAM:

            team_map[track_id] = 0

        else:

            team_map[track_id] = 1

    return team_map