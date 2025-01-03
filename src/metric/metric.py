"""Module to compute the disguise metric."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config.settings import get_settings
from ..core.positions import convert_ints_to_positions
from ..io.datasets import CSVReader, CSVWriter
from ..utils.data_processing import remove_unwanted_columns

RELEVANT_COLUMNS_FRAME_DISGUISE: list[str] = [
    "gameId",
    "playId",
    "nflId",
    "jerseyNumber",
    "club",
    "position",
    "displayName",
    "frameId",
    "time",
    "wasInitialPassRusher",
    "blitz_probability",
    "frame_disguise",
    "num_frames",
]

RELEVANT_COLUMNS_PLAY_DISGUISE: list[str] = [
    "gameId",
    "playId",
    "nflId",
    "jerseyNumber",
    "position",
    "displayName",
    "club",
    "wasInitialPassRusher",
    "play_disguise",
    "quarterbackHit",
    "causedPressure",
    "tackleForALoss",
]


def compute_frame_disguise_metric(df: pd.DataFrame) -> pd.Series[float]:
    """Compute the disguise metric for each frame.

    The disguise metric is a measure of how well a pass rusher disguises their blitz.
    It is computed as the absolute value of the difference between the predicted
    blitz probability and the 'wasInitialPassRusher' flag.

    A higher value indicates a better disguise, while a lower value indicates a
    worse disguise.

    Args:
        df (pd.DataFrame): Dataframe with the following columns:
            - 'blitz_probability': predicted blitz probability.
            - 'wasInitialPassRusher': flag indicating whether the player was an
                initial pass rusher.

    Returns:
        pd.Series[float]: Disguise metric.
    """
    return (df["blitz_probability"] - df["wasInitialPassRusher"]).abs()


def compute_play_disguise_metric(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the play disguise metric.

    The play disguise metric is a measure of how well a player disguises
    their intention (blitz vs coverage) in a play.
    It is computed as the sum of the frame disguise metric for all frames
    in the play divided by the number of frames in the play.

    Args:
        df (pd.DataFrame): Dataframe with the following columns:
            - 'gameId': game identifier.
            - 'playId': play identifier.
            - 'nflId': player identifier.
            - 'frameId': frame identifier.
            - 'frame_disguise': frame disguise metric.
            - 'num_frames': number of frames in play.

    Returns:
        pd.Series[float]: Play disguise metric.
    """
    grouped = df.groupby(["gameId", "playId", "nflId"])

    metric = (grouped["frame_disguise"].sum() / grouped["num_frames"].first()).astype(
        "float32"
    )
    result = metric.reset_index(name="play_disguise")

    merged_df = pd.merge(df, result, on=["gameId", "playId", "nflId"], how="left")

    return merged_df.drop_duplicates(subset=["gameId", "playId", "nflId"])


def compute_number_of_frames_in_play(df: pd.DataFrame) -> pd.Series[int]:
    """Compute the number of frames in each play.

    Args:
        df (pd.DataFrame): Dataframe with the following columns:
            - 'gameId': game identifier.
            - 'playId': play identifier.
            - 'nflId': player identifier.
            - 'frameId': frame identifier.

    Returns:
        pd.DataFrame: Dataframe with the number of frames in each play.
    """
    return df.groupby(["gameId", "playId", "nflId"])["frameId"].transform("count")


def transform_columns_frame_disguise(df: pd.DataFrame) -> pd.DataFrame:
    """Transform the columns of the dataframe.

    Downcast the columns to the smallest possible data type.

    Args:
        df (pd.DataFrame): Dataframe to transform.

    Returns:
        pd.DataFrame: Transformed dataframe.
    """
    df["gameId"] = pd.to_numeric(df["gameId"], downcast="integer")
    df["playId"] = pd.to_numeric(df["playId"], downcast="integer")
    df["nflId"] = pd.to_numeric(df["nflId"], downcast="integer")
    df["frameId"] = pd.to_numeric(df["frameId"], downcast="integer")
    df["position"] = df["position"].astype("category")
    df["displayName"] = df["displayName"].astype("category")
    df["club"] = df["club"].astype("category")

    df["jerseyNumber"] = pd.to_numeric(df["jerseyNumber"], downcast="integer")
    df["wasInitialPassRusher"] = pd.to_numeric(
        df["wasInitialPassRusher"], downcast="integer"
    )
    df["blitz_probability"] = pd.to_numeric(df["blitz_probability"], downcast="float")
    df["frame_disguise"] = pd.to_numeric(df["frame_disguise"], downcast="float")
    df["num_frames"] = pd.to_numeric(df["num_frames"], downcast="integer")

    return df


def transform_columns_play_disguise(df: pd.DataFrame) -> pd.DataFrame:
    """Transform the columns to the smallest possible data type.

    Args:
        df (pd.DataFrame): Dataframe to transform.

    Returns:
        pd.DataFrame: Transformed dataframe.
    """
    df["gameId"] = pd.to_numeric(df["gameId"], downcast="integer")
    df["playId"] = pd.to_numeric(df["playId"], downcast="integer")
    df["nflId"] = pd.to_numeric(df["nflId"], downcast="integer")
    df["jerseyNumber"] = pd.to_numeric(df["jerseyNumber"], downcast="integer")
    df["position"] = df["position"].astype("category")
    df["displayName"] = df["displayName"].astype("category")
    df["club"] = df["club"].astype("category")

    df["wasInitialPassRusher"] = pd.to_numeric(
        df["wasInitialPassRusher"], downcast="integer"
    )
    df["play_disguise"] = pd.to_numeric(df["play_disguise"], downcast="float")

    return df


def compute_weighted_disguise_score(df: pd.DataFrame, k: float = 0.4) -> pd.DataFrame:
    """Compute the weighted disguise score.

    The weighted disguise score is a measure of how well a player disguises
    their intention (blitz vs coverage) in a play.
    Frames that are closer to the snap have a higher weight.
    The weighted disguise score is computed as follows:
    - The frame disguise is weighted by a factor of exp(-k * time_to_snap)
    - The weighted frame disguise is summed for all frames in the play
    - The weighted frame disguise is divided by total weight for all frames in the play

    Args:
        df (pd.DataFrame): Dataframe with the following columns:
            - 'gameId': game identifier.
            - 'playId': play identifier.
            - 'nflId': player identifier.
            - 'frameId': frame identifier.
            - 'frame_disguise': frame disguise metric.
            - 'num_frames': number of frames in play.
        k (float, optional): Decay factor. Defaults to 0.4.

    Returns:
        pd.DataFrame: Dataframe with the weighted disguise score.
    """
    df["time"] = df["time"].apply(lambda x: x if "." in x else x + ".000")
    df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S.%f")
    df["snap_time"] = df.groupby(["gameId", "playId", "nflId"])["time"].transform("max")
    df["time_to_snap"] = (df["snap_time"] - df["time"]).dt.total_seconds()

    df["weight"] = np.exp(-k * df["time_to_snap"])

    df["weighted_frame_disguise"] = df["frame_disguise"] * df["weight"]
    df["total_disguise"] = df.groupby(["gameId", "playId", "nflId"])[
        "weighted_frame_disguise"
    ].transform("sum")

    df["sum_weights"] = df.groupby(["gameId", "playId", "nflId"])["weight"].transform(
        "sum"
    )
    df["disguise_score"] = df["total_disguise"] / df["sum_weights"]

    df = df.drop_duplicates(subset=["gameId", "playId", "nflId"])

    df = df[
        [
            "gameId",
            "playId",
            "nflId",
            "jerseyNumber",
            "club",
            "position",
            "displayName",
            "wasInitialPassRusher",
            "disguise_score",
        ]
    ]

    return df


def main() -> None:
    """Compute the disguise metric and save the results."""
    settings = get_settings()

    reader = CSVReader()
    writer = CSVWriter()

    blitz_probability_data = reader.read(settings.BLITZ_PROBABILITY_RESULTS_FILE)

    player_plays = reader.read(
        settings.get_data_file_path(file=settings.PLAYER_PLAYS_FILE)
    )
    player_plays["gameId"] = player_plays["gameId"].astype(int)
    player_plays["playId"] = player_plays["playId"].astype(int)
    player_plays["nflId"] = player_plays["nflId"].astype(int)

    blitz_probability_data["frame_disguise"] = compute_frame_disguise_metric(
        blitz_probability_data
    )
    blitz_probability_data["num_frames"] = compute_number_of_frames_in_play(
        blitz_probability_data
    )
    blitz_probability_data = remove_unwanted_columns(
        blitz_probability_data,
        [
            col
            for col in blitz_probability_data.columns
            if col not in RELEVANT_COLUMNS_FRAME_DISGUISE
        ],
    )
    blitz_probability_data = transform_columns_frame_disguise(blitz_probability_data)

    writer.write(
        settings.FRAME_DISGUISE_RESULTS_FILE,
        blitz_probability_data[RELEVANT_COLUMNS_FRAME_DISGUISE],
    )

    weighted_disguise_score = compute_weighted_disguise_score(blitz_probability_data)
    weighted_disguise_score = weighted_disguise_score.merge(
        player_plays[
            [
                "gameId",
                "playId",
                "nflId",
                "quarterbackHit",
                "causedPressure",
                "tackleForALoss",
            ]
        ],
        how="left",
        on=["gameId", "playId", "nflId"],
    )
    weighted_disguise_score = convert_ints_to_positions(
        weighted_disguise_score, ["position"]
    )

    writer.write(
        settings.WEIGHTED_PLAY_DISGUISE_RESULTS_FILE,
        weighted_disguise_score,
    )

    blitz_probability_data = compute_play_disguise_metric(blitz_probability_data)

    blitz_probability_data = remove_unwanted_columns(
        blitz_probability_data,
        [
            col
            for col in blitz_probability_data.columns
            if col not in RELEVANT_COLUMNS_PLAY_DISGUISE
        ],
    )
    blitz_probability_data = transform_columns_play_disguise(blitz_probability_data)

    blitz_probability_data = blitz_probability_data.merge(
        player_plays[
            [
                "gameId",
                "playId",
                "nflId",
                "quarterbackHit",
                "causedPressure",
                "tackleForALoss",
            ]
        ],
        how="left",
        on=["gameId", "playId", "nflId"],
    )

    blitz_probability_data = convert_ints_to_positions(
        blitz_probability_data, ["position"]
    )

    writer.write(
        settings.PLAY_DISGUISE_RESULTS_FILE,
        blitz_probability_data[RELEVANT_COLUMNS_PLAY_DISGUISE],
    )


if __name__ == "__main__":
    main()
