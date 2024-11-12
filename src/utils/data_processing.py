"""Utility module for processing data."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def merge_dataframes(dataframes: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge multiple dataframes into a single dataframe.

    Args:
        dataframes (list[pd.DataFrame]): List of dataframes to merge.

    Returns:
        pd.DataFrame: Merged dataframe.
    """
    logger.info(f"Merging {len(dataframes)} dataframes")

    return pd.concat(dataframes, axis=0, ignore_index=True)


def remove_unwanted_columns(
    dataframe: pd.DataFrame, columns: str | list[str]
) -> pd.DataFrame:
    """Remove one or more columns from a DataFrame.

    Args:
        dataframe (pd.DataFrame): DataFrame to remove columns from.
        columns (str | list[str]): Column name or list of column names to remove.

    Raises:
        ValueError: If any of the specified columns do not exist in the DataFrame.

    Returns:
        pd.DataFrame: DataFrame with the specified columns removed.
    """
    if isinstance(columns, str):
        columns = [columns]

    logger.info(f"Attempting to remove columns: {columns}")

    for column in columns:
        if column not in dataframe.columns:
            logger.error(f"Column '{column}' does not exist in DataFrame")
            raise ValueError(f"Column '{column}' does not exist in DataFrame")

    logger.info(f"Successfully removed columns: {columns}")
    return dataframe.drop(columns, axis=1)
