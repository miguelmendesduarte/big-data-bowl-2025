"""Color palette for visualization."""

from enum import StrEnum


class TeamColor(StrEnum):
    """Enumeration of NFL team colors."""

    ARI = "#97233F"
    ATL = "#A71930"
    BAL = "#241773"
    BUF = "#00338D"
    CAR = "#0085CA"
    CHI = "#C83803"
    CIN = "#FB4F14"
    CLE = "#311D00"
    DAL = "#003594"
    DEN = "#FB4F14"
    DET = "#0076B6"
    GB = "#203731"
    HOU = "#03202F"
    IND = "#002C5F"
    JAX = "#9F792C"
    KC = "#E31837"
    LA = "#FFA300"
    LAC = "#0080C6"
    LV = "#000000"
    MIA = "#008E97"
    MIN = "#4F2683"
    NE = "#002244"
    NO = "#D3BC8D"
    NYG = "#0B2265"
    NYJ = "#125740"
    PHI = "#004C54"
    PIT = "#FFB612"
    SEA = "#69BE28"
    SF = "#AA0000"
    TB = "#D50A0A"
    TEN = "#4B92DB"
    WAS = "#5A1414"


BALL_COLOR: str = "#CBB67C"


def get_team_color(team: str) -> str:
    """Get the hex color code for a given NFL team.

    Args:
        team (str): NFL team abbreviation.

    Raises:
        ValueError: Invalid team abbreviation.

    Returns:
        str: Hex color code.
    """
    team = team.upper()
    try:
        return TeamColor[team].value
    except KeyError as e:
        raise ValueError(f"Invalid team abbreviation: '{team}'") from e
