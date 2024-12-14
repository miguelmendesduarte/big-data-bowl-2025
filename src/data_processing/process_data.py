"""Module with the pipeline for data processing."""

import gc
import logging

import pandas as pd
from tqdm import tqdm

from ..config.logs import configure_logging
from ..config.settings import get_settings
from ..io.datasets import CSVReader, CSVWriter
from .cleaning.plays import clean_plays_data
from .cleaning.tracking import clean_tracking_data
from .feature_engineering.features import add_features

logger = logging.getLogger(__name__)


def process_data(
    tracking_df: pd.DataFrame,
    plays_df: pd.DataFrame,
    players_df: pd.DataFrame,
    player_plays_df: pd.DataFrame,
    games_df: pd.DataFrame,
) -> pd.DataFrame:
    """Clean and process play and tracking data.

    Clean play and tracking data, and add all features.

    Args:
        tracking_df (pd.DataFrame): Dataframe with tracking data.
        plays_df (pd.DataFrame): Dataframe with play data.
        players_df (pd.DataFrame): Dataframe with player data.
        player_plays_df (pd.DataFrame): Dataframe with player play data.
        games_df (pd.DataFrame): Dataframe with game data.

    Returns:
        pd.DataFrame: Processed dataframe with all features.
    """
    cleaned_plays_df = clean_plays_data(plays_df)
    cleaned_tracking_df = clean_tracking_data(
        plays_data=cleaned_plays_df, tracking_data=tracking_df
    )

    return add_features(
        cleaned_tracking_df, cleaned_plays_df, players_df, player_plays_df, games_df
    )


def main() -> None:
    """Process all data in the pipeline and save the processed data."""
    settings = get_settings()
    configure_logging()

    reader = CSVReader()
    writer = CSVWriter()

    weeks = range(1, 10)

    games_file = settings.get_data_file_path(file=settings.GAMES_FILE)
    plays_file = settings.get_data_file_path(file=settings.PLAYS_FILE)
    players_file = settings.get_data_file_path(file=settings.PLAYERS_FILE)
    player_plays_file = settings.get_data_file_path(file=settings.PLAYER_PLAYS_FILE)
    tracking_files = {}
    for week in weeks:
        tracking_files[f"week_{week}"] = settings.get_tracking_file_path(week=week)

    games_data = reader.read(games_file)
    plays_data = reader.read(plays_file)
    players_data = reader.read(players_file)
    player_plays_data = reader.read(player_plays_file)

    for week in tqdm(weeks, desc="Processing weeks"):
        tracking_data = reader.read(tracking_files[f"week_{week}"])

        processed_tracking_data: pd.DataFrame = process_data(
            tracking_df=tracking_data,
            plays_df=plays_data,
            players_df=players_data,
            player_plays_df=player_plays_data,
            games_df=games_data,
        )

        writer.write(
            settings.get_tracking_file_path(week=week, processed=True),
            data=processed_tracking_data,
        )

        del tracking_data, processed_tracking_data
        gc.collect()


if __name__ == "__main__":
    main()
