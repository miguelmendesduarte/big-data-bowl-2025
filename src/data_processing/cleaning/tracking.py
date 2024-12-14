"""Module for cleaning tracking data."""

import logging

import numpy as np
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


def remove_pre_line_set_frames(tracking_data: pd.DataFrame) -> pd.DataFrame:
    """Remove pre-line set frames for each play and player.

    This is used for inference. There is no need to compute the
    probability of blitzing when teams are not set yet.

    Args:
        tracking_data (pd.DataFrame): Tracking data.

    Returns:
        pd.DataFrame: Tracking data without pre-line set frames.
    """
    line_set_frames = (
        tracking_data[tracking_data.event == "line_set"]
        .groupby(["gameId", "playId", "nflId"])["frameId"]
        .min()
        .reset_index(name="line_set_frameId")
    )

    tracking_data = tracking_data.merge(
        line_set_frames, on=["gameId", "playId", "nflId"], how="left"
    )
    tracking_data_post_line_set = tracking_data[
        tracking_data.frameId >= tracking_data.line_set_frameId
    ]

    tracking_data_post_line_set = tracking_data_post_line_set.drop(
        ["line_set_frameId"], axis=1
    )

    return tracking_data_post_line_set


def remove_frames_before_ball_snap(
    tracking_data: pd.DataFrame, num_frames: int = 15
) -> pd.DataFrame:
    """Remove 'num_frames' before the ball snap for each play and player.

    ATTENTION:
    This is used for training. The model should not be trained on the
    frames long before the ball snap.

    Args:
        tracking_data (pd.DataFrame): Tracking data.
        num_frames (int, optional): Number of frames to remove before the ball snap.
            Defaults to 15.

    Returns:
        pd.DataFrame: Tracking data without 'num_frames' before the ball snap.
    """
    ball_snap_frames = (
        tracking_data[tracking_data["event"] == "ball_snap"]
        .groupby(["gameId", "playId", "nflId"])["frameId"]
        .min()
        .reset_index(name="ball_snap_frameId")
    )

    tracking_data = tracking_data.merge(
        ball_snap_frames, on=["gameId", "playId", "nflId"], how="left"
    )
    tracking_data_post_ball_snap = tracking_data[
        tracking_data.frameId >= (tracking_data.ball_snap_frameId - num_frames)
    ]

    tracking_data_post_ball_snap = tracking_data_post_ball_snap.drop(
        ["ball_snap_frameId"], axis=1
    )

    return tracking_data_post_ball_snap


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


def rotate_angles(tracking_data: pd.DataFrame) -> pd.DataFrame:
    """Rotate direction and orientation angles.

    Now, 0º points to the right, and increases counterclockwise.

    Args:
        tracking_data (pd.DataFrame): Tracking data.

    Returns:
        pd.DataFrame: Tracking data with rotated angles.
    """
    logger.info("Rotating direction and orientation angles")
    tracking_data["o"] = -(tracking_data["o"] - 90) % 360
    tracking_data["dir"] = -(tracking_data["dir"] - 90) % 360

    return tracking_data


def assign_speed_sign(tracking_data: pd.DataFrame) -> pd.DataFrame:
    """Modify 's' based on the player's direction relative to the LOS.

    **ATTENTION**: Use after "rotate_angles"!

    If the player is towards the line of scrimmage, the speed is positive.
    If the player is away from the line of scrimmage, the speed is negative.

    Args:
        tracking_data (pd.DataFrame): Tracking data.

    Returns:
        pd.DataFrame: Tracking data with 's' modified.
    """
    towards_line_of_scrimmage = (tracking_data["dir"] > 90) & (
        tracking_data["dir"] < 270
    )

    tracking_data["s"] = np.where(
        towards_line_of_scrimmage, tracking_data["s"], -tracking_data["s"]
    )

    return tracking_data


def clean_tracking_data(
    plays_data: pd.DataFrame, tracking_data: pd.DataFrame
) -> pd.DataFrame:
    """Clean the tracking data.

    Applies the following cleaning steps:
    - Remove plays not in the plays data
    - Remove post-snap frames
    - Convert plays left to right
    - Rotate angles
    - Assign speed sign
    - Remove frames before line set

    Args:
        plays_data (pd.DataFrame): Dataframe with plays.
        tracking_data (pd.DataFrame): Dataframe with tracking data.

    Returns:
        pd.DataFrame: Cleaned tracking data.
    """
    logger.info("Cleaning tracking data...")
    cleaned_tracking_data = (
        tracking_data.pipe(filter_plays_in_tracking_data, plays_data)
        .pipe(remove_post_snap_frames)
        .pipe(remove_pre_line_set_frames)
        .pipe(convert_plays_left_to_right)
        .pipe(rotate_angles)
        .pipe(assign_speed_sign)
    )
    logger.info(f"Cleaned tracking data, {len(cleaned_tracking_data)} rows remain")

    return cleaned_tracking_data
