"""Module for creating the training and testing datasets."""

import pandas as pd
from tqdm import tqdm

from ...config.settings import get_settings
from ...config.training_settings import get_training_settings
from ...core.positions import convert_positions_to_int, get_only_defensive_backs
from ...io.datasets import CSVReader, CSVWriter
from ...utils.data_processing import drop_duplicate_rows_tracking, merge_dataframes
from ..cleaning.tracking import remove_frames_before_ball_snap


def fill_nan_values_in_target_column(
    dataframe: pd.DataFrame, target_column: str = "wasInitialPassRusher"
) -> pd.DataFrame:
    """Fill NaN values in the target column with 0.

    Args:
        dataframe (pd.DataFrame): Tracking data.
        target_column (str, optional): Column to fill.
            Defaults to "wasInitialPassRusher".

    Returns:
        pd.DataFrame: Tracking data with NaN values filled.
    """
    return dataframe.fillna({target_column: 0})


def create_datasets() -> None:
    """Create and save the training and testing datasets."""
    global_settings = get_settings()
    training_settings = get_training_settings()

    reader = CSVReader()
    writer = CSVWriter()

    weeks = range(1, 10)  # 9 weeks

    processed_tracking_files = {}
    for week in weeks:
        processed_tracking_files[f"week_{week}"] = (
            global_settings.get_tracking_file_path(week=week, processed=True)
        )

    training_dataframes = []
    testing_dataframes = []
    for week in tqdm(weeks, desc="Processing weeks"):
        processed_tracking_data = reader.read(processed_tracking_files[f"week_{week}"])

        processed_tracking_data = (
            processed_tracking_data.pipe(remove_frames_before_ball_snap)
            .pipe(drop_duplicate_rows_tracking)
            .pipe(get_only_defensive_backs)
            .pipe(convert_positions_to_int)
            .pipe(fill_nan_values_in_target_column)
        )

        if week <= training_settings.NUM_TRAINING_WEEKS:
            training_dataframes.append(processed_tracking_data)
        else:
            testing_dataframes.append(processed_tracking_data)

    training_dataframe = merge_dataframes(training_dataframes)
    testing_dataframe = merge_dataframes(testing_dataframes)

    writer.write(training_settings.TRAIN_FILE, training_dataframe)
    writer.write(training_settings.TEST_FILE, testing_dataframe)


if __name__ == "__main__":
    create_datasets()
