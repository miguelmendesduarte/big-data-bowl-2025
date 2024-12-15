#!/bin/bash

MLFLOW_PORT=${MLFLOW_PORT:-5000}  # Default port 5000 if not set
MLFLOW_HOST=${MLFLOW_HOST:-0.0.0.0}  # Default to 0.0.0.0 if not set

MLFLOW_UI_CMD="mlflow ui --host $MLFLOW_HOST --port $MLFLOW_PORT"

echo "Starting MLflow UI at http://$MLFLOW_HOST:$MLFLOW_PORT"

$MLFLOW_UI_CMD
