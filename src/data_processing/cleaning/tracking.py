"""Module for cleaning tracking data."""

import logging

import pandas as pd

from ...visualization.field import FIELD_LENGTH, FIELD_WIDTH

logger = logging.getLogger(__name__)


def filter_plays_in_tracking_data(
    plays_data: pd.DataFrame, tracking_data: pd.DataFrame
) -> pd.DataFrame:
    """Remove unwanted plays from tracking data based on plays data.

    Args:
        plays_data (pd.DataFrame): Plays data.
        tracking_data (pd.DataFrame): Tracking data.

    Returns:
        pd.DataFrame: Tracking data with only the plays present in the plays data.
    """
    filtered_tracking_data = tracking_data.merge(
        plays_data, on=["gameId", "playId"], how="inner"
    )
    filtered_tracking_data = filtered_tracking_data[tracking_data.columns]
    logger.info(f"Filtered {len(tracking_data) - len(filtered_tracking_data)} plays")

    return filtered_tracking_data


def remove_post_snap_frames(tracking_data: pd.DataFrame) -> pd.DataFrame:
    """Remove post-snap frames.

    Only keep snap and pre-snap frames.

    Args:
        tracking_data (pd.DataFrame): Tracking data.

    Returns:
        pd.DataFrame: Tracking data without post-snap frames.
    """
    logger.info("Removing post-snap frames")
    return tracking_data[tracking_data.frameType != "AFTER_SNAP"]


def convert_plays_left_to_right(tracking_data: pd.DataFrame) -> pd.DataFrame:
    """Flip tracking data where plays are right to left.

    Args:
        tracking_data (pd.DataFrame): Tracking data.

    Returns:
        pd.DataFrame: Tracking data where all plays are left to right.
    """
    right_to_left_plays = tracking_data[tracking_data.playDirection == "left"].copy()

    right_to_left_plays["x"] = FIELD_LENGTH - right_to_left_plays["x"]
    right_to_left_plays["y"] = FIELD_WIDTH - right_to_left_plays["y"]
    right_to_left_plays["o"] = (right_to_left_plays["o"] + 180) % 360
    right_to_left_plays["dir"] = (right_to_left_plays["dir"] + 180) % 360

    tracking_data.loc[right_to_left_plays.index, ["x", "y", "o", "dir"]] = (
        right_to_left_plays[["x", "y", "o", "dir"]]
    )
    logger.info(f"Flipped {len(right_to_left_plays)} right to left plays")

    return tracking_data
