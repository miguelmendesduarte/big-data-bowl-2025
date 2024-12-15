"""Module for evaluating machine learning models."""

from __future__ import annotations

from typing import Protocol

import pandas as pd
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class Model(Protocol):
    """Protocol for machine learning models.

    For type checking purposes.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series[int]) -> None:
        """Train model on given features and target values.

        Args:
            X (pd.DataFrame): Features.
            y (pd.Series[int]): Target values.
        """
        ...

    def predict(self, X: pd.DataFrame) -> pd.Series[int]:
        """Predict target values for given features.

        Args:
            X (pd.DataFrame): Features.

        Returns:
            pd.Series[int]: Predicted target values.
        """
        ...


def evaluate_model(
    model: Model, test_data: tuple[pd.DataFrame, pd.Series[int]]
) -> dict[str, float | pd.DataFrame]:
    """Evaluate a machine learning model on test data.

    Args:
        model (Model): Trained model with fit and predict methods.
        test_data (tuple[pd.DataFrame, pd.Series]): Test features and target values.

    Returns:
        dict[str, float]: Evaluation metrics.
            Includes accuracy, precision, recall, F1 score and confusion matrix.
    """
    X_test, y_test = test_data

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="binary", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="binary", zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average="binary"),
        "confusion_matrix": pd.DataFrame(
            confusion_matrix(y_test, y_pred),
            index=["Actual_0", "Actual_1"],
            columns=["Predicted_0", "Predicted_1"],
        ),
    }

    return metrics
