"""Module to add new features to the dataset."""

# Motion metrics VS situation metrics - will he rush VS does it make sense to rush

from __future__ import annotations

from enum import StrEnum

import numpy as np
import pandas as pd


def merge_player_position_with_tracking_data(
    player_df: pd.DataFrame, tracking_df: pd.DataFrame, on_column: str = "nflId"
) -> pd.DataFrame:
    """Merge player position with tracking data.

    Args:
        player_df (pd.DataFrame): Dataframe with player data.
        tracking_df (pd.DataFrame): Dataframe with tracking data.
        on_column (str, optional): Column to merge on.
            Defaults to "nflId".

    Returns:
        pd.DataFrame: Merged dataframe.
    """
    return player_df[["nflId", "position"]].merge(tracking_df, on=on_column)


def add_pass_rusher_label_to_tracking_data(
    tracking_df: pd.DataFrame, player_play_df: pd.DataFrame
) -> pd.DataFrame:
    """Add pass rusher label to tracking data.

    Based on 'wasInitialPassRusher' variable in player play data.

    Args:
        tracking_df (pd.DataFrame): Dataframe with tracking data.
        player_play_df (pd.DataFrame): Dataframe with player play data.

    Returns:
        pd.DataFrame: Dataframe with pass rusher label added.
    """
    return tracking_df.merge(
        player_play_df[["nflId", "playId", "gameId", "wasInitialPassRusher"]],
        on=["nflId", "playId", "gameId"],
        how="left",
    )


def get_distance_to_qb(tracking_df: pd.DataFrame) -> pd.DataFrame:
    """Get the euclidean distance from each player to the quarterback.

    Args:
        tracking_df (pd.DataFrame): Dataframe with tracking data.

    Returns:
        pd.DataFrame: Dataframe with distance to quarterback added.
    """
    qb_df = tracking_df[tracking_df.position == "QB"][
        ["gameId", "playId", "frameId", "x", "y"]
    ]

    merged_df = tracking_df.merge(
        qb_df, on=["gameId", "playId", "frameId"], how="left", suffixes=("", "_qb")
    )

    merged_df["distance_to_qb"] = (
        (merged_df.x - merged_df.x_qb) ** 2 + (merged_df.y - merged_df.y_qb) ** 2
    ) ** 0.5

    merged_df = merged_df.drop(columns=["x_qb", "y_qb"])

    return merged_df


def get_orientation_difference_to_qb(tracking_df: pd.DataFrame) -> pd.DataFrame:
    """Get difference between each player's orientation and angle to the quarterback.

    The angle to the quarterback is calculated using 'arctan2' function.

    Args:
        tracking_df (pd.DataFrame): Dataframe with tracking data.

    Returns:
        pd.DataFrame: Dataframe with orientation difference to quarterback added.
    """
    qb_df = tracking_df[tracking_df.position == "QB"][
        ["gameId", "playId", "frameId", "x", "y"]
    ]

    merged_df = tracking_df.merge(
        qb_df, on=["gameId", "playId", "frameId"], how="left", suffixes=("", "_qb")
    )

    merged_df["dx"] = merged_df["x"] - merged_df["x_qb"]
    merged_df["dy"] = merged_df["y"] - merged_df["y_qb"]

    angle_rad = np.arctan2(merged_df["dy"], merged_df["dx"])
    angle_deg = np.degrees(angle_rad)
    angle_deg = angle_deg + 180  # Defensive players are on the right

    merged_df["orientation_to_qb"] = merged_df["o"] - angle_deg

    merged_df = merged_df.drop(columns=["x_qb", "y_qb", "dx", "dy"])

    return merged_df


def get_direction_difference_to_qb(tracking_df: pd.DataFrame) -> pd.DataFrame:
    """Get the difference between each player's direction and angle to the quarterback.

    The angle to the quarterback is calculated using 'arctan2' function.

    Args:
        tracking_df (pd.DataFrame): Dataframe with tracking data.

    Returns:
        pd.DataFrame: Dataframe with direction difference to quarterback added.
    """
    qb_df = tracking_df[tracking_df.position == "QB"][
        ["gameId", "playId", "frameId", "x", "y"]
    ]

    merged_df = tracking_df.merge(
        qb_df, on=["gameId", "playId", "frameId"], how="left", suffixes=("", "_qb")
    )

    merged_df["dx"] = merged_df["x"] - merged_df["x_qb"]
    merged_df["dy"] = merged_df["y"] - merged_df["y_qb"]

    angle_rad = np.arctan2(merged_df["dy"], merged_df["dx"])
    angle_deg = np.degrees(angle_rad)
    angle_deg = angle_deg + 180  # Defensive players are on the right

    merged_df["direction_to_qb"] = merged_df["dir"] - angle_deg

    merged_df = merged_df.drop(columns=["x_qb", "y_qb", "dx", "dy"])

    return merged_df


def get_distance_to_line_of_scrimmage(
    tracking_df: pd.DataFrame, plays_df: pd.DataFrame
) -> pd.DataFrame:
    """Get the euclidean distance from each player to the line of scrimmage.

    Args:
        tracking_df (pd.DataFrame): Dataframe with tracking data.
        plays_df (pd.DataFrame): Dataframe with plays data.

    Returns:
        pd.DataFrame: Dataframe with distance to line of scrimmage added.
    """
    merged_df = plays_df[["gameId", "playId", "absoluteYardlineNumber"]].merge(
        tracking_df, on=["gameId", "playId"]
    )

    mask = merged_df["playDirection"] == "left"
    merged_df.loc[mask, "absoluteYardlineNumber"] = (
        120 - merged_df["absoluteYardlineNumber"]
    )

    merged_df["distance_to_line_of_scrimmage"] = abs(
        merged_df.absoluteYardlineNumber - merged_df.x
    )

    merged_df = merged_df.drop(columns=["absoluteYardlineNumber"])

    return merged_df


def _compute_closest_opponent_distance(tracking_df: pd.DataFrame) -> pd.Series[float]:
    """Compute the euclidean distance between each player in the same frame.

    Args:
        tracking_df (pd.DataFrame): Dataframe with tracking data.

    Returns:
        pd.Series[float]: Series with distances to closest opponent.
    """
    x_coords = tracking_df["x"].to_numpy()
    y_coords = tracking_df["y"].to_numpy()
    teams = tracking_df["club"].to_numpy()

    positions = np.stack((x_coords, y_coords), axis=1)
    pairwise_distances = np.linalg.norm(
        positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=2
    )

    np.fill_diagonal(pairwise_distances, np.inf)  # Avoid selecting self as closest

    for i, team in enumerate(teams):
        pairwise_distances[i, teams == team] = (
            np.inf
        )  # Avoid selecting teammates as closest

    closest_distances = pairwise_distances.min(axis=1)

    return pd.Series(closest_distances, dtype=float, index=tracking_df.index)


def get_distance_to_closest_opponent(tracking_df: pd.DataFrame) -> pd.DataFrame:
    """Get the euclidean distance from each player to the closest opponent.

    Args:
        tracking_df (pd.DataFrame): Dataframe with tracking data.

    Returns:
        pd.DataFrame: Dataframe with distance to closest opponent added.
    """
    tracking_df["distance_to_closest_opponent"] = (
        tracking_df.groupby(["gameId", "playId", "frameId"])
        .apply(_compute_closest_opponent_distance, include_groups=False)
        .reset_index(drop=True, level=[0, 1, 2])
    )

    return tracking_df


def _compute_closest_opponent_position(tracking_df: pd.DataFrame) -> pd.Series[float]:
    """Compute the closest opponent position for each player in the same frame.

    Args:
        tracking_df (pd.DataFrame): Dataframe with tracking data.

    Returns:
        pd.Series[float]: Series with closest opponent positions.
    """
    x_coords = tracking_df["x"].to_numpy()
    y_coords = tracking_df["y"].to_numpy()
    teams = tracking_df["club"].to_numpy()

    positions = tracking_df["position"]

    player_positions = np.stack((x_coords, y_coords), axis=1)
    pairwise_distances = np.linalg.norm(
        player_positions[:, np.newaxis, :] - player_positions[np.newaxis, :, :], axis=2
    )

    np.fill_diagonal(pairwise_distances, np.inf)  # Avoid selecting self as closest

    # Avoid teammates from being selected as the closest opponent
    for i, team in enumerate(teams):
        pairwise_distances[i, teams == team] = np.inf

    closest_indices = pairwise_distances.argmin(axis=1)
    closest_positions = positions.iloc[closest_indices]

    return pd.Series(closest_positions.values, dtype=float, index=tracking_df.index)


def get_position_of_closest_opponent(tracking_df: pd.DataFrame) -> pd.DataFrame:
    """Get the position of the closest opponent for each player in the same frame.

    Args:
        tracking_df (pd.DataFrame): Dataframe with tracking data.

    Returns:
        pd.DataFrame: Dataframe with position of closest opponent added.
    """
    tracking_df["position_of_closest_opponent"] = (
        tracking_df.groupby(["gameId", "playId", "frameId"])
        .apply(_compute_closest_opponent_position, include_groups=False)
        .reset_index(drop=True, level=[0, 1, 2])
    )

    return tracking_df


def _compute_closest_opponent_orientation(
    tracking_df: pd.DataFrame,
) -> pd.Series[float]:
    """Compute the closest opponent orientation for each player in the same frame.

    Args:
        tracking_df (pd.DataFrame): Dataframe with tracking data.

    Returns:
        pd.Series[float]: Series with closest opponent orientations.
    """
    x_coords = tracking_df["x"].to_numpy()
    y_coords = tracking_df["y"].to_numpy()
    teams = tracking_df["club"].to_numpy()

    orientations = tracking_df["o"]

    player_positions = np.stack((x_coords, y_coords), axis=1)
    pairwise_distances = np.linalg.norm(
        player_positions[:, np.newaxis, :] - player_positions[np.newaxis, :, :], axis=2
    )

    np.fill_diagonal(pairwise_distances, np.inf)  # Avoid selecting self as closest

    # Avoid teammates from being selected as the closest opponent
    for i, team in enumerate(teams):
        pairwise_distances[i, teams == team] = np.inf

    closest_indices = pairwise_distances.argmin(axis=1)
    closest_orientations = orientations.iloc[closest_indices]

    return pd.Series(closest_orientations.values, dtype=float, index=tracking_df.index)


def get_orientation_of_closest_opponent(tracking_df: pd.DataFrame) -> pd.DataFrame:
    """Get the orientation of the closest opponent for each player in the same frame.

    Args:
        tracking_df (pd.DataFrame): Dataframe with tracking data.

    Returns:
        pd.DataFrame: Dataframe with orientation of closest opponent added.
    """
    # Not sure if it is helpful
    tracking_df["orientation_of_closest_opponent"] = (
        tracking_df.groupby(["gameId", "playId", "frameId"])
        .apply(_compute_closest_opponent_orientation, include_groups=False)
        .reset_index(drop=True, level=[0, 1, 2])
    )

    return tracking_df


def get_down(tracking_df: pd.DataFrame, plays_df: pd.DataFrame) -> pd.DataFrame:
    """Get down for each play in tracking data.

    Args:
        tracking_df (pd.DataFrame): Dataframe with tracking data.
        plays_df (pd.DataFrame): Dataframe with plays data.

    Returns:
        pd.DataFrame: Dataframe with down added.
    """
    return plays_df[["gameId", "playId", "down"]].merge(
        tracking_df, on=["gameId", "playId"]
    )


def get_yards_to_go(tracking_df: pd.DataFrame, plays_df: pd.DataFrame) -> pd.DataFrame:
    """Get yards to go for each play in tracking data.

    Args:
        tracking_df (pd.DataFrame): Dataframe with tracking data.
        plays_df (pd.DataFrame): Dataframe with plays data.

    Returns:
        pd.DataFrame: Dataframe with yards to go added.
    """
    return plays_df[["gameId", "playId", "yardsToGo"]].merge(
        tracking_df, on=["gameId", "playId"]
    )


def get_yards_to_endzone(
    tracking_df: pd.DataFrame, plays_df: pd.DataFrame
) -> pd.DataFrame:
    """Get yards to endzone for each play in tracking data.

    Endzone is at x=110 and all plays are left to right.

    Args:
        tracking_df (pd.DataFrame): Dataframe with tracking data.
        plays_df (pd.DataFrame): Dataframe with plays data.

    Returns:
        pd.DataFrame: Dataframe with yards to endzone added.
    """
    merged_df = plays_df[["gameId", "playId", "absoluteYardlineNumber"]].merge(
        tracking_df, on=["gameId", "playId"]
    )

    mask = merged_df["playDirection"] == "left"
    merged_df.loc[mask, "absoluteYardlineNumber"] = (
        120 - merged_df["absoluteYardlineNumber"]
    )

    merged_df["yardsToEndzone"] = 110 - merged_df.absoluteYardlineNumber

    merged_df = merged_df.drop(columns=["absoluteYardlineNumber"])

    return merged_df


def get_score_differential(
    tracking_df: pd.DataFrame, plays_df: pd.DataFrame, games_df: pd.DataFrame
) -> pd.DataFrame:
    """Get the pre-snap score differential from the perspective of the defensive team.

    If the defense is playing as the away team, the differential is
    calculated as the opponent's score minus the defensive team's score, and vice versa
    for the home team.

    Args:
        tracking_df (pd.DataFrame): Dataframe with tracking data.
        plays_df (pd.DataFrame): Dataframe with play data.
        games_df (pd.DataFrame): Dataframe with game data.

    Returns:
        pd.DataFrame: Dataframe with score differential added.
    """
    merged_df = plays_df[
        ["gameId", "playId", "defensiveTeam", "preSnapHomeScore", "preSnapVisitorScore"]
    ].merge(tracking_df, on=["gameId", "playId"])

    merged_df = merged_df.merge(
        games_df[["gameId", "homeTeamAbbr", "visitorTeamAbbr"]], on="gameId"
    )

    is_defense_visitor = merged_df["defensiveTeam"] == merged_df["visitorTeamAbbr"]

    merged_df["scoreDifferential"] = (
        merged_df["preSnapVisitorScore"] - merged_df["preSnapHomeScore"]
    ) * is_defense_visitor + (
        merged_df["preSnapHomeScore"] - merged_df["preSnapVisitorScore"]
    ) * ~is_defense_visitor

    merged_df = merged_df.drop(
        columns=[
            "defensiveTeam",
            "preSnapHomeScore",
            "preSnapVisitorScore",
            "homeTeamAbbr",
            "visitorTeamAbbr",
        ]
    )

    return merged_df


def get_time_remaining_in_seconds(
    tracking_df: pd.DataFrame, plays_df: pd.DataFrame
) -> pd.DataFrame:
    """Get the the remaining time in seconds for each play.

    The remaining time is based on the 'gameClock' column (formatted as 'MM:SS')
    and the 'quarter' column.

    Args:
        tracking_df (pd.DataFrame): Dataframe with tracking data.
        plays_df (pd.DataFrame): Dataframe with play data.

    Returns:
        pd.DataFrame: Dataframe with time remaining in seconds added.
    """
    merged_df = plays_df[["gameId", "playId", "gameClock", "quarter"]].merge(
        tracking_df, on=["gameId", "playId"]
    )

    time_parts = merged_df["gameClock"].str.split(":", expand=True)

    merged_df["timeRemainingInSeconds"] = time_parts[0].astype(int) * 60 + time_parts[
        1
    ].astype(int)

    seconds_per_quarter = 15 * 60
    merged_df["timeRemainingInSeconds"] += (
        4 - merged_df["quarter"]
    ) * seconds_per_quarter

    merged_df = merged_df.drop(columns=["gameClock", "quarter"])

    return merged_df


class RelevantPosition(StrEnum):
    """Enumeration for player positions relevant to blocking assignments.

    The positions included are:
        - RB (Running Back)
        - FB (Fullback)
        - TE (Tight End)

    These positions are typically involved in blocking rushers.
    """

    RB = "RB"
    FB = "FB"
    TE = "TE"


class Side(StrEnum):
    """Side of the field based on the quarterback position."""

    left = "left"
    right = "right"


def get_number_of_players(
    tracking_df: pd.DataFrame, position: RelevantPosition, side: Side
) -> pd.Series[int] | pd.DataFrame:
    """Get the number of players of a given position and side of the field.

    Args:
        tracking_df (pd.DataFrame): Dataframe with tracking data.
        position (RelevantPosition): Position of the player.
        side (Side): Side of the field based on the quarterback position.

    Raises:
        ValueError: If the side is invalid.

    Returns:
        pd.Series: Series with number of players of given position and side.
    """
    qb_df = tracking_df[tracking_df["position"] == "QB"][
        ["gameId", "playId", "frameId", "y"]
    ]

    merged_df = tracking_df.merge(
        qb_df, on=["gameId", "playId", "frameId"], how="left", suffixes=("", "_qb")
    )

    position_df = merged_df[merged_df["position"] == position.value]

    if side == Side.left:
        relevant_df = position_df[position_df["y"] >= position_df["y_qb"]]
    elif side == Side.right:
        relevant_df = position_df[position_df["y"] < position_df["y_qb"]]
    else:
        raise ValueError(f"Invalid side: {side}")

    count_series = relevant_df.groupby(["gameId", "playId", "frameId"]).size()

    count_series.name = f"{position.value}_{side}_count"

    aligned_series = count_series.reindex(
        tracking_df.set_index(["gameId", "playId", "frameId"]).index, fill_value=0
    )
    aligned_series = aligned_series.reset_index(drop=True)

    return aligned_series


def get_TEs_on_right(tracking_df: pd.DataFrame) -> pd.DataFrame:
    """Get the number of tight ends on the right side of the QB.

    Args:
        tracking_df (pd.DataFrame): Dataframe with tracking data.

    Returns:
        pd.DataFrame: Dataframe with number of tight ends on the right added.
    """
    tracking_df["TEs_on_right"] = get_number_of_players(
        tracking_df, RelevantPosition.TE, Side.right
    )

    return tracking_df


def get_TEs_on_left(tracking_df: pd.DataFrame) -> pd.DataFrame:
    """Get the number of tight ends on the left side of the QB.

    Args:
        tracking_df (pd.DataFrame): Dataframe with tracking data.

    Returns:
        pd.DataFrame: Dataframe with number of tight ends on the left added.
    """
    tracking_df["TEs_on_left"] = get_number_of_players(
        tracking_df, RelevantPosition.TE, Side.left
    )

    return tracking_df


def get_RBs_on_right(tracking_df: pd.DataFrame) -> pd.DataFrame:
    """Get the number of running backs on the right side of the QB.

    Args:
        tracking_df (pd.DataFrame): Dataframe with tracking data.

    Returns:
        pd.DataFrame: Dataframe with number of running backs on the right added.
    """
    tracking_df["RBs_on_right"] = get_number_of_players(
        tracking_df, RelevantPosition.RB, Side.right
    )

    return tracking_df


def get_RBs_on_left(tracking_df: pd.DataFrame) -> pd.DataFrame:
    """Get the number of running backs on the left side of the QB.

    Args:
        tracking_df (pd.DataFrame): Dataframe with tracking data.

    Returns:
        pd.DataFrame: Dataframe with number of running backs on the left added.
    """
    tracking_df["RBs_on_left"] = get_number_of_players(
        tracking_df, RelevantPosition.RB, Side.left
    )

    return tracking_df


def get_FBs_on_right(tracking_df: pd.DataFrame) -> pd.DataFrame:
    """Get the number of fullbacks on the right side of the QB.

    Args:
        tracking_df (pd.DataFrame): Dataframe with tracking data.

    Returns:
        pd.DataFrame: Dataframe with number of fullbacks on the right added.
    """
    tracking_df["FBs_on_right"] = get_number_of_players(
        tracking_df, RelevantPosition.FB, Side.right
    )

    return tracking_df


def get_FBs_on_left(tracking_df: pd.DataFrame) -> pd.DataFrame:
    """Get the number of fullbacks on the left side of the QB.

    Args:
        tracking_df (pd.DataFrame): Dataframe with tracking data.

    Returns:
        pd.DataFrame: Dataframe with number of fullbacks on the left added.
    """
    tracking_df["FBs_on_left"] = get_number_of_players(
        tracking_df, RelevantPosition.FB, Side.left
    )

    return tracking_df


def get_defenders_near_LOS(
    tracking_df: pd.DataFrame,
    plays_df: pd.DataFrame,
    near_LOS_depth: int = 2,
    near_LOS_width: int = 7,
) -> pd.DataFrame:
    """Get defenders near LOS for each play in tracking data.

    A defender is considered near the line of scrimmage if they are within:
        - 'near_LOS_depth' yards of the quarterback
        - 'near_LOS_width' yards of the line of scrimmage

    Args:
        tracking_df (pd.DataFrame): Dataframe with tracking data.
        plays_df (pd.DataFrame): Dataframe with plays data.
        near_LOS_depth (int, optional): Maximum 'y' distance to quarterback.
            Defaults to 2.
        near_LOS_width (int, optional): Maximum 'x' distance to line of scrimmage.
            Defaults to 7.

    Returns:
        pd.DataFrame: Dataframe with defenders near LOS added.
    """
    merged_df = plays_df[
        ["gameId", "playId", "absoluteYardlineNumber", "defensiveTeam"]
    ].merge(tracking_df, on=["gameId", "playId"])

    mask = merged_df["playDirection"] == "left"
    merged_df.loc[mask, "absoluteYardlineNumber"] = (
        120 - merged_df["absoluteYardlineNumber"]
    )

    qb_df = tracking_df[tracking_df["position"] == "QB"][
        ["gameId", "playId", "frameId", "y"]
    ]

    merged_df = merged_df.merge(
        qb_df, on=["gameId", "playId", "frameId"], how="left", suffixes=("", "_qb")
    )

    defenders_df = merged_df[merged_df["club"] == merged_df["defensiveTeam"]].copy()

    defenders_df["defenders_near_LOS"] = (
        (defenders_df["absoluteYardlineNumber"] >= defenders_df["x"] - near_LOS_depth)
        & (defenders_df["y"] <= defenders_df["y_qb"] + near_LOS_width)
        & (defenders_df["y"] >= defenders_df["y_qb"] - near_LOS_width)
    ).astype(int)

    defenders_count_df = (
        defenders_df.groupby(["gameId", "playId", "frameId"])["defenders_near_LOS"]
        .sum()
        .reset_index()
    )

    merged_df = merged_df.merge(
        defenders_count_df, on=["gameId", "playId", "frameId"], how="left"
    )

    merged_df = merged_df.drop(
        columns=["y_qb", "absoluteYardlineNumber", "defensiveTeam"]
    )

    return merged_df
