"""Module for creating the inference dataset."""

from tqdm import tqdm

from ...config.settings import get_settings
from ...config.training_settings import get_training_settings
from ...core.positions import convert_positions_to_int, get_only_defensive_backs
from ...io.datasets import CSVReader, CSVWriter
from ...utils.data_processing import drop_duplicate_rows_tracking, merge_dataframes
from ..cleaning.tracking import remove_pre_line_set_frames
from ..training.datasets import fill_nan_values_in_target_column


def create_inference_dataset() -> None:
    """Create and save the inference dataset."""
    global_settings = get_settings()
    training_settings = get_training_settings()

    reader = CSVReader()
    writer = CSVWriter()

    inference_weeks = range(training_settings.NUM_TRAINING_WEEKS + 1, 10)

    processed_tracking_files = {}
    for week in inference_weeks:
        processed_tracking_files[f"week_{week}"] = (
            global_settings.get_tracking_file_path(week=week, processed=True)
        )

    inference_dataframes = []
    for week in tqdm(inference_weeks, desc="Processing weeks"):
        processed_tracking_data = reader.read(processed_tracking_files[f"week_{week}"])

        processed_tracking_data = (
            processed_tracking_data.pipe(remove_pre_line_set_frames)
            .pipe(drop_duplicate_rows_tracking)
            .pipe(get_only_defensive_backs)
            .pipe(convert_positions_to_int)
            .pipe(fill_nan_values_in_target_column)
        )

        inference_dataframes.append(processed_tracking_data)

    inference_dataframe = merge_dataframes(inference_dataframes)

    writer.write(
        global_settings.INFERENCE_FILE,
        inference_dataframe,
    )


if __name__ == "__main__":
    create_inference_dataset()
