"""Module for creating and getting a machine learning model."""

from __future__ import annotations

from itertools import product
from typing import Generator

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from ..config.training_settings import ModelType, get_training_settings
from .evaluation import Model


def get_model_configs() -> Generator[dict[str, int | float], None, None]:
    """Get all model configurations based on the hyperparameter grid.

    Yields:
        Generator[dict[str, int | float], None, None]: Model configurations.
    """
    training_settings = get_training_settings()
    hyperparameter_grid = training_settings.HYPERPARAMETER_GRID

    keys, values = zip(*hyperparameter_grid.items())
    for combination in product(*values):
        yield dict(zip(keys, combination))


def get_model(
    y_train: pd.Series[int], hyperparameters: dict[str, int | float]
) -> Model:
    """Get a machine learning model based on the given hyperparameters.

    Args:
        y_train (pd.Series[int]): Target values.
        hyperparameters (dict[str, int  |  float]): Hyperparameters.

    Raises:
        ValueError: Unknown model type.

    Returns:
        Model: Machine learning model.
    """
    training_settings = get_training_settings()

    random_state = training_settings.RANDOM_SEED

    negative_class_count = np.bincount(y_train.astype(int))[0]
    positive_class_count = np.bincount(y_train.astype(int))[1]

    scale_pos_weight = (
        (negative_class_count / positive_class_count)
        if positive_class_count != 0
        else 1
    )

    if training_settings.MODEL == ModelType.XGB:
        return XGBClassifier(
            random_state=random_state,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            **hyperparameters,
        )

    raise ValueError(f"Unknown model type: {training_settings.MODEL}")
