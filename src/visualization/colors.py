"""Color palette for visualization."""

from ..config.settings import TEAMS

# Primary and secondary colors for each team
TEAM_COLORS: dict[str, tuple[str, str]] = {
    "ARI": ("#97233F", "#000000"),
    "ATL": ("#A71930", "#A71930"),
    "BAL": ("#241773", "#F2C464"),
    "BUF": ("#00338D", "#B0B7BC"),
    "CAR": ("#0085CA", "#4688F6"),
    "CHI": ("#C83803", "#C83803"),
    "CIN": ("#FB4F14", "#FFD700"),
    "CLE": ("#311D00", "#FFC80A"),
    "DAL": ("#003594", "#002244"),
    "DEN": ("#FB4F14", "#FFC080"),
    "DET": ("#0076B6", "#B2B2B2"),
    "GB": ("#203731", "#FFD700"),
    "HOU": ("#03202F", "#B0B7BC"),
    "IND": ("#002C5F", "#FFC80A"),
    "JAX": ("#9F792C", "#9F792C"),
    "KC": ("#E31837", "#FFD700"),
    "LA": ("#FFA300", "#8E24AA"),
    "LAC": ("#0080C6", "#FFC80A"),
    "LV": ("#000000", "#B0B7BC"),
    "MIA": ("#008E97", "#FFC080"),
    "MIN": ("#4F2683", "#FFC80A"),
    "NE": ("#002244", "#C83803"),
    "NO": ("#D3BC8D", "#D3BC8D"),
    "NYG": ("#0B2265", "#8E24AA"),
    "NYJ": ("#125740", "#125740"),
    "PHI": ("#004C54", "#FFC080"),
    "PIT": ("#FFB612", "#FFB612"),
    "SEA": ("#69BE28", "#69BE28"),
    "SF": ("#AA0000", "#AA0000"),
    "TB": ("#D50A0A", "#D50A0A"),
    "TEN": ("#4B92DB", "#4B92DB"),
    "WAS": ("#5A1414", "#5A1414"),
}


# Constants for other colors
BALL_COLOR = "#CBB67C"
FIELD_COLOR = "#3B7A57"
LINE_COLOR = "#FFFFFF"


def get_team_colors(team: str) -> tuple[str, str]:
    """Get the hex color codes for a given NFL team.

    Args:
        team (str): NFL team abbreviation.

    Raises:
        ValueError: If the provided team abbreviation is invalid.

    Returns:
        tuple (str, str): Hex color codes - (primary, secondary).
    """
    team = team.upper()
    if team in TEAMS:
        return TEAM_COLORS[team]
    else:
        raise ValueError(f"Invalid team abbreviation: '{team}'")
