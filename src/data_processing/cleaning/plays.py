"""Module for cleaning plays data."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def remove_plays_with_penalty(plays_data: pd.DataFrame) -> pd.DataFrame:
    """Remove plays with penalty.

    Args:
        plays_data (pd.DataFrame): Dataframe with plays.

    Returns:
        pd.DataFrame: Dataframe without plays with penalty.
    """
    cleaned_data = plays_data[plays_data.playNullifiedByPenalty != "Y"]
    logger.info(f"Removed {len(plays_data) - len(cleaned_data)} plays with penalty")

    return cleaned_data


def remove_non_passing_plays(plays_data: pd.DataFrame) -> pd.DataFrame:
    """Remove non-passing plays.

    Args:
        plays_data (pd.DataFrame): Dataframe with plays.

    Returns:
        pd.DataFrame: Dataframe without non-passing plays.
    """
    cleaned_data = plays_data[plays_data.isDropback]
    logger.info(f"Removed {len(plays_data) - len(cleaned_data)} non-passing plays")

    return cleaned_data


def remove_wildcat_formation_plays(plays_data: pd.DataFrame) -> pd.DataFrame:
    """Remove wildcat offense formation plays.

    In wildcat offense formation, the ball is not snapped to the quarterback, but
    to a player of another position lined up at the quarterback position.

    Args:
        plays_data (pd.DataFrame): Dataframe with plays.

    Returns:
        pd.DataFrame: Dataframe without wildcat offense formation plays.
    """
    cleaned_data = plays_data[plays_data.offenseFormation != "WILDCAT"]
    logger.info(
        f"Removed {len(plays_data) - len(cleaned_data)} wildcat offense formation plays"
    )

    return cleaned_data


def remove_designed_rollouts_and_runs(plays_data: pd.DataFrame) -> pd.DataFrame:
    """Remove designed rollouts and runs.

    For rollouts, the quarterback is usually meant to either throw the ball
    after getting to the edge or, if the option is there, run it themselves.

    Designed runs focus on the quarterback or running back carrying the ball,
    with blocking schemes tailored to support that.

    Args:
        plays_data (pd.DataFrame): Dataframe with plays.

    Returns:
        pd.DataFrame: Dataframe without designed rollouts and runs.
    """
    cleaned_data = plays_data[
        ~plays_data.dropbackType.isin(
            ["DESIGNED_RUN", "DESIGNED_ROLLOUT_LEFT", "DESIGNED_ROLLOUT_RIGHT"]
        )
    ]
    logger.info(
        f"Removed {len(plays_data) - len(cleaned_data)} designed rollouts and runs"
    )

    return cleaned_data
