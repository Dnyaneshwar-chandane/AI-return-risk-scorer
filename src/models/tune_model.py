import pathlib
import sys
import yaml
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


def main():

    curr_dir = pathlib.Path(__file__)

    home_dir = (
        curr_dir
        .parent
        .parent
        .parent
    )

    # -------------------------
    # Load parameters
    # -------------------------

    params_file = home_dir / "params.yaml"

    with open(params_file) as f:
        params = yaml.safe_load(f)

    C_values = params["tuning"]["C"]

    # -------------------------
    # Load data
    # -------------------------

    input_path = home_dir / sys.argv[1]

    X_train = pd.read_csv(
        input_path / "X_train_trf.csv"
    )

    train_data = pd.read_csv(
        input_path / "selected_train.csv"
    )

    y_train = train_data["returned"]

    # -------------------------
    # Base model
    # -------------------------

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    # -------------------------
    # Grid
    # -------------------------

    param_grid = {
        "C": C_values
    }

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )

    # -------------------------
    # MLflow
    # -------------------------

    mlflow.set_experiment(
        "logistic_regression_tuning"
    )

    with mlflow.start_run(
        run_name="LR_GridSearch"
    ):

        grid.fit(
            X_train,
            y_train
        )

        # Best parameters
        best_C = grid.best_params_["C"]
        best_score = grid.best_score_

        # Log results
        mlflow.log_param(
            "best_C",
            best_C
        )

        mlflow.log_param(
            "cv",
            5
        )

        mlflow.log_param(
            "scoring",
            "f1"
        )

        mlflow.log_metric(
            "best_cv_f1",
            best_score
        )

        # Log best model
        mlflow.sklearn.log_model(
            grid.best_estimator_,
            name="model"
        )

        print("\nBest Parameters:")
        print(grid.best_params_)

        print(
            f"\nBest CV F1: "
            f"{best_score:.4f}"
        )


if __name__ == "__main__":
    main()