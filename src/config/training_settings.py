"""Machine learning training settings."""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

from src.config.settings import get_settings

settings = get_settings()


TRAIN_FILENAME = "train.csv"
TEST_FILENAME = "test.csv"
SEED = 42
MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT_NAME = "nfl_big_data_bowl"


class ModelType(StrEnum):
    """Model type."""

    XGB = "xgboost"


class TrainingSettings(BaseSettings):
    """Training settings."""

    # Directories
    TRAIN_DIR: Path = settings.DATA_DIR / "train"
    TEST_DIR: Path = settings.DATA_DIR / "test"

    # Files
    TRAIN_FILE: Path = Field(
        default=Path(TRAIN_DIR / TRAIN_FILENAME),
        description="Path to training data file.",
    )
    TEST_FILE: Path = Field(
        default=Path(TEST_DIR / TEST_FILENAME), description="Path to test data file."
    )

    # Data
    NUM_TRAINING_WEEKS: int = Field(
        default=5, description="Number of weeks to use for training the model."
    )  # Testing will be done on the remaining weeks

    # Model configuration
    MODEL: ModelType = Field(
        default=ModelType.XGB, description="Type of machine learning model."
    )
    HYPERPARAMETER_GRID: dict[str, list[int | float]] = Field(
        default={
            "n_estimators": [100, 150, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7, 9],
            "min_child_weight": [1, 3, 5],
        },
        description="Grid of hyperparameters to search over.",
    )
    ALL_FEATURES: list[str] = Field(
        default=[
            "yardsToGo",
            "down",
            "position",
            "x",
            "y",
            "s",  # Feature with neg. permutation importance
            "a",
            "dis",
            "o",
            "dir",  # Feature with neg. permutation importance
            "distance_to_qb",
            "orientation_to_qb",
            "direction_to_qb",  # Feature with neg. permutation importance
            "distance_to_line_of_scrimmage",
            "yardsToEndzone",  # Feature with neg. permutation importance
            "scoreDifferential",
            "timeRemainingInSeconds",
            "defenders_near_LOS",
            "TEs_on_right",
            "TEs_on_left",
            "FBs_on_right",
            "FBs_on_left",
            "RBs_on_right",
            "RBs_on_left",
            "distance_to_closest_opponent",  # Feature with neg. permutation importance
            "position_of_closest_opponent",
            "orientation_to_closest_opponent",
        ],
        description="List of features to use for training.",
    )
    TRAINING_FEATURES: list[str] = Field(
        default=[
            "yardsToGo",
            "down",
            "position",
            "x",
            "y",
            "a",
            "dis",
            "o",
            "distance_to_qb",
            "orientation_to_qb",
            "distance_to_line_of_scrimmage",
            "scoreDifferential",
            "timeRemainingInSeconds",
            "defenders_near_LOS",
            "TEs_on_right",
            "TEs_on_left",
            "FBs_on_right",
            "FBs_on_left",
            "RBs_on_right",
            "RBs_on_left",
            "position_of_closest_opponent",
            "orientation_to_closest_opponent",
        ],
        description="List of features to use for training.",
    )
    TARGET: str = Field(
        default="wasInitialPassRusher", description="Target variable to predict."
    )

    # Reproducibility
    RANDOM_SEED: int = Field(
        default=SEED, description="Random seed for reproducibility."
    )

    # MLFlow configuration
    MLFLOW_TRACKING_URI: str = Field(
        default=MLFLOW_TRACKING_URI, description="MLFlow tracking server URI."
    )
    MLFLOW_EXPERIMENT_NAME: str = Field(
        default=MLFLOW_EXPERIMENT_NAME, description="MLFlow experiment name."
    )
    MLFLOW_RUN_NAME: str = Field(
        default="nfl_big_data_bowl", description="MLFlow run name."
    )
    LOG_MODEL: bool = Field(
        default=True, description="Whether to log the model to MLFlow."
    )


@lru_cache(maxsize=1)
def get_training_settings() -> TrainingSettings:
    """Get training settings.

    Returns:
        TrainingSettings: Settings for training a machine learning model.
    """
    return TrainingSettings()
