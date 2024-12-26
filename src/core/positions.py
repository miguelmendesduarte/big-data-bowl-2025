"""Positions module."""

import pandas as pd

DB_POSITIONS: list[str] = ["CB", "DB", "FS", "SS"]

position_mapping = {
    "WR": 1,
    "CB": 2,
    "T": 3,
    "G": 4,
    "OLB": 5,
    "DE": 6,
    "ILB": 7,
    "FS": 8,
    "DT": 9,
    "TE": 10,
    "C": 11,
    "RB": 12,
    "QB": 13,
    "SS": 14,
    "NT": 15,
    "MLB": 16,
    "FB": 17,
    "LB": 18,
    "DB": 19,
}


def get_only_defensive_backs(tracking_data: pd.DataFrame) -> pd.DataFrame:
    """Get only defensive backs from the tracking data.

    Args:
        tracking_data (pd.DataFrame): Tracking data.

    Returns:
        pd.DataFrame: Tracking data with only defensive backs.
    """
    return tracking_data[tracking_data.position.isin(DB_POSITIONS)]


def convert_positions_to_int(
    tracking_data: pd.DataFrame,
    columns: list[str] = ["position", "position_of_closest_opponent"],
) -> pd.DataFrame:
    """Convert multiple position columns to integer.

    Possible columns:
    - "position"
    - "position_of_closest_opponent"

    Args:
        tracking_data (pd.DataFrame): Tracking data.
        columns (list[str]): Columns with positions.

    Returns:
        pd.DataFrame: _description_
    """
    for column in columns:
        tracking_data[column] = (
            tracking_data[column].map(position_mapping).astype("int8")
        )
    return tracking_data


def convert_ints_to_positions(
    tracking_data: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Convert integer positions to string positions.

    Args:
        tracking_data (pd.DataFrame): Dataframe with tracking data.
        columns (list[str]): Columns with integer positions.

    Returns:
        pd.DataFrame: Dataframe with string positions.
    """
    reverse_position_mapping = {v: k for k, v in position_mapping.items()}

    for column in columns:
        tracking_data[column] = tracking_data[column].map(reverse_position_mapping)

    return tracking_data
