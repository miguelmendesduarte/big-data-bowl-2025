"""Module for cleaning plays data."""

import logging

import pandas as pd

from ...utils.data_processing import remove_unwanted_columns

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


def remove_QB_kneel_plays(plays_data: pd.DataFrame) -> pd.DataFrame:
    """Remove plays where quarterback kneels.

    Args:
        plays_data (pd.DataFrame): Dataframe with plays.

    Returns:
        pd.DataFrame: Dataframe without plays where quarterback kneels.
    """
    cleaned_data = plays_data[plays_data.qbKneel != 1]
    logger.info(f"Removed {len(plays_data) - len(cleaned_data)} qb kneel plays")

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


def clean_plays_data(plays_data: pd.DataFrame) -> pd.DataFrame:
    """Clean plays data.

    Applies the following cleaning steps:
    - Remove non-passing plays
    - Remove plays with penalty
    - Remove designed rollouts and runs
    - Remove wildcat offense formation plays

    Args:
        plays_data (pd.DataFrame): Dataframe with plays.

    Returns:
        pd.DataFrame: Dataframe with cleaned plays.
    """
    return (
        plays_data.pipe(remove_non_passing_plays)
        .pipe(remove_QB_kneel_plays)
        .pipe(remove_plays_with_penalty)
        .pipe(remove_designed_rollouts_and_runs)
        .pipe(remove_wildcat_formation_plays)
        .pipe(
            remove_unwanted_columns,
            [
                "playDescription",
                "yardlineSide",
                "yardlineNumber",
                "playNullifiedByPenalty",
                "preSnapHomeTeamWinProbability",
                "preSnapVisitorTeamWinProbability",
                "expectedPoints",
                "offenseFormation",
                "receiverAlignment",
                "playClockAtSnap",
                "passResult",
                "passLength",
                "targetX",
                "targetY",
                "playAction",
                "dropbackType",
                "dropbackDistance",
                "passLocationType",
                "timeToThrow",
                "timeInTackleBox",
                "timeToSack",
                "passTippedAtLine",
                "unblockedPressure",
                "qbSpike",
                "qbKneel",
                "qbSneak",
                "rushLocationType",
                "penaltyYards",
                "prePenaltyYardsGained",
                "yardsGained",
                "homeTeamWinProbabilityAdded",
                "visitorTeamWinProbilityAdded",
                "expectedPointsAdded",
                "isDropback",
                "pff_runConceptPrimary",
                "pff_runConceptSecondary",
                "pff_runPassOption",
                "pff_passCoverage",
                "pff_manZone",
            ],
        )
    )
