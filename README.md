# Disguised Intentions: How DBs Keep Offenses Guessing

This repository supports my submission to the **NFL Big Data Bowl 2025** submission, [Disguised Intentions](https://www.kaggle.com/code/miguelmd123/disguised-intentions). The project quantifies how effectively defensive backs (DBs) disguise their intentions pre-snap and how this impacts defensive outcomes.

## Prerequisites

To set up the project locally,

- Ensure you have **Poetry** installed (check [this link](https://python-poetry.org/docs/#installation) for more details).

- Ensure you have a compatible **Python** version (see `pyproject.toml` for supported versions).

## Installation

1. Clone the repository to your desired folder:

    ```bash
    git clone git@github.com:miguelmendesduarte/big-data-bowl-2025.git <desired-folder-name>
    ```

2. Enter the desired folder:

    ```bash
    cd <desired-folder-name>
    ```

3. Install the dependencies:

    ```bash
    poetry install
    ```

4. Initiate the virtual environment:

    ```bash
    poetry shell
    ```

## Running

1. **Download the NFL Big Data Bowl 2025 data**:

    Download the data from [this link](https://www.kaggle.com/competitions/nfl-big-data-bowl-2025/data) and save it in the `data/raw/` directory.

2. **Process raw data**:

    Run the following command to created processed datasets:

    ```bash
    python -m src.data_processing.process_data
    ```

    This will store the processed data in the `data/processed/` directory.

3. **Prepare train and test datasets**:

    Run this command to prepare the train and test datasets:

    ```bash
    python -m src.data_processing.training.datasets
    ```

    The resulting datasets are divided into:

    - **Train**: Weeks 1 to 5 (stored in `data/train/`)
    - **Test**: Weeks 6 to 9 (stored in `data/test/`)

4. **Train the model**:

    - Start the MLflow UI to track experiments and results:

        ```bash
        sh scripts/start_mlflow_ui.sh
        ```

    - Adjust hyperparameters in `src/config/training_settings.py` under the `HYPERPARAMETER_GRID` variable. The current options include:

        ```json
        "n_estimators": [100, 150, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7, 9],
        "min_child_weight": [1, 3, 5]
        ```

    - Train the model with:

        ```bash
        python -m src.training.train
        ```

    - In the MLflow UI, review results and identify the best-performing hyperparameter combination based on log loss. For this project, the optimal settings are:

        - `max_depth`: 9
        - `n_estimators`: 300
        - `min_child_weight`: 3
        - `learning_rate`: 0.1

5. **Update the model settings**:

    Once the best model is identified, update the `src/config/settings.py` file with the experiment ID and run ID of the model to be used.

6. **Create the inference dataset**:

    To generate the dataset for inference, run:

    ```bash
    python -m src.data_processing.inference.dataset
    ```

    This will create the `inference.csv` dataset, which will be used to obtain the blitz probability results.

7. **Get blitz probability predictions**:

    Use the trained model to predict blitz probabilities by running:

    ```bash
    python -m src.inference.predictions
    ```

    This will generate `blitz_probability_results.csv` inside the `data/inference/` directory.

8. **Compute disguise scores**:

    Calculate the disguise scores by running:

    ```bash
    python -m src.metric.metric
    ```

    This will create two files inside the `data/metric/` directory:

    - `play_disguise_results.csv`: Average frame disguise scores (not used in the submission).

    - `weighted_play_disguise_results.csv`: Weighted average of frame results (used in the submission).

## Results

Once you have obtained the `weighted_play_disguise_results.csv`, it can be used to test hypotheses and plot graphs based on the **Disguise Score** metric.

## Notes

- This project uses **MLflow** for experiment tracking. If you're unfamiliar with MLflow, you can refer to the [official documentation](https://www.mlflow.org/#core-concepts) for more information on how to use the UI and manage experiments.

- The dataset used for this project comes from the **NFL Big Data Bowl 2025**. Ensure you have access to the data before proceeding with the steps above.

- You may adjust the hyperparameters based on experimentation and your preferences.

## Contact

I'm available for any questions or clarifications. Feel free to reach out! :smile:
