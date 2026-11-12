import xml.etree.ElementTree as ET


def load_track_teams(xml_path):

    team_map = {}

    tree = ET.parse(xml_path)

    root = tree.getroot()

    players = root.find(".//Players")

    for player in players:

        player_id = int(
            player.attrib["id"]
        )

        team_id = player.attrib["teamId"]

        if "o4Hossvb8M" in team_id:

            team_map[player_id] = 0

        else:

            team_map[player_id] = 1

    return team_map