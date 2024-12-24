"""Main module for training machine learning models."""

from datetime import datetime

import mlflow

from ..config.training_settings import get_training_settings
from ..io.datasets import CSVReader
from .evaluation import evaluate_model
from .model import get_model, get_model_configs


def main() -> None:
    """Train machine learning models.

    The testing results will be logged to MLFlow.
    The one with the best score will be logged as the best model and
    used to predict the probability of blitz.
    """
    settings = get_training_settings()

    reader = CSVReader()

    train_data = reader.read(settings.TRAIN_FILE)
    test_data = reader.read(settings.TEST_FILE)

    X_train, y_train = (
        train_data[settings.TRAINING_FEATURES],
        train_data[settings.TARGET],
    )
    X_test, y_test = test_data[settings.TRAINING_FEATURES], test_data[settings.TARGET]

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

    for config in get_model_configs():
        run_name = (
            f"{settings.MLFLOW_RUN_NAME}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )

        with mlflow.start_run(run_name=run_name):
            model = get_model(y_train, config)

            model.fit(X_train, y_train)

            metrics = evaluate_model(model, (X_test, y_test))

            mlflow.log_params(config)
            mlflow.log_metrics(metrics)

            if settings.LOG_MODEL:
                mlflow.sklearn.log_model(
                    model, artifact_path="model", input_example=X_train
                )


if __name__ == "__main__":
    main()
