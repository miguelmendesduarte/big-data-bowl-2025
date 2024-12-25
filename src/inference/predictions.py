"""Inference pipeline."""

from __future__ import annotations

import mlflow
import mlflow.sklearn
import pandas as pd

from ..config.settings import get_settings
from ..config.training_settings import get_training_settings
from ..io.datasets import CSVReader, CSVWriter


def predict_blitz_probability(features: pd.DataFrame) -> pd.Series[float]:
    """Predict the probability of blitz for the given features.

    Args:
        features (pd.DataFrame): Features.

    Returns:
        pd.Series[float]: Predicted probabilities.
    """
    global_settings = get_settings()

    model = mlflow.sklearn.load_model(f"file://{global_settings.MODEL_PATH}")

    results: pd.Series[float] = model.predict_proba(features)[:, 1]

    return results


def main() -> None:
    """Predict the probability of blitz for the given features.

    The results will be saved to the results file.
    """
    global_settings = get_settings()
    training_settings = get_training_settings()

    reader = CSVReader()
    writer = CSVWriter()

    inference_data = reader.read(global_settings.INFERENCE_FILE)

    features = inference_data[training_settings.TRAINING_FEATURES]

    probabilities = predict_blitz_probability(features)
    inference_data["blitz_probability"] = probabilities

    writer.write(global_settings.RESULTS_FILE, inference_data)


if __name__ == "__main__":
    main()
