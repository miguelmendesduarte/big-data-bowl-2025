"""Color palette for visualization."""

from ..config.settings import TEAMS

# Primary and secondary colors for each team
TEAM_COLORS: dict[str, tuple[str, str]] = {
    "ARI": ("#97233F", "#FFB612"),
    "ATL": ("#A71930", "#000000"),
    "BAL": ("#241773", "#9E7C0C"),
    "BUF": ("#00338D", "#C60C30"),
    "CAR": ("#0085CA", "#101820"),
    "CHI": ("#0B162A", "#C83803"),
    "CIN": ("#FB4F14", "#000000"),
    "CLE": ("#311D00", "#FF3C00"),
    "DAL": ("#003594", "#869397"),
    "DEN": ("#FB4F14", "#002244"),
    "DET": ("#0076B6", "#B0B7BC"),
    "GB": ("#203731", "#FFB612"),
    "HOU": ("#03202F", "#A71930"),
    "IND": ("#002C5F", "#B0B7BC"),
    "JAX": ("#006778", "#D7A22A"),
    "KC": ("#E31837", "#FFB81C"),
    "LA": ("#003594", "#FFD100"),
    "LAC": ("#0080C6", "#FFC20E"),
    "LV": ("#000000", "#A5ACAF"),
    "MIA": ("#008E97", "#FC4C02"),
    "MIN": ("#4F2683", "#FFC62F"),
    "NE": ("#002244", "#C60C30"),
    "NO": ("#D3BC8D", "#101820"),
    "NYG": ("#0B2265", "#A71930"),
    "NYJ": ("#125740", "#000000"),
    "PHI": ("#004C54", "#A5ACAF"),
    "PIT": ("#FFB612", "#101820"),
    "SEA": ("#002244", "#69BE28"),
    "SF": ("#AA0000", "#B3995D"),
    "TB": ("#D50A0A", "#0A0A08"),
    "TEN": ("#0C2340", "#4B92DB"),
    "WAS": ("#773141", "#FFB612"),
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
