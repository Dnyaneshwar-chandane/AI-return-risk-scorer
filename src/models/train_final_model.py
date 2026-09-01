import pathlib
import sys
import yaml
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


def main():

    curr_dir = pathlib.Path(__file__)

    home_dir = (
        curr_dir
        .parent
        .parent
        .parent
    )

    # Parameters
    with open(home_dir / "params.yaml") as f:
        params = yaml.safe_load(f)["final_model"]

    # Data
    input_path = home_dir / sys.argv[1]

    X_train = pd.read_csv(
        input_path / "X_train_trf.csv"
    )

    X_test = pd.read_csv(
        input_path / "X_test_trf.csv"
    )

    train_data = pd.read_csv(
        input_path / "selected_train.csv"
    )

    test_data = pd.read_csv(
        input_path / "selected_test.csv"
    )

    y_train = train_data["returned"]
    y_test = test_data["returned"]

    # Final model
    model = LogisticRegression(
        **params,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )
    model_path = home_dir / "models"

    model_path.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        model_path / "return_risk_model.pkl"
    )

    # Test prediction
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = {
        "precision_class_1": precision_score(
            y_test, y_pred
        ),
        "recall_class_1": recall_score(
            y_test, y_pred
        ),
        "f1_class_1": f1_score(
            y_test, y_pred
        ),
        "roc_auc": roc_auc_score(
            y_test, y_prob
        ),
        "pr_auc": average_precision_score(
            y_test, y_prob
        )
    }

    # MLflow
    mlflow.set_experiment(
        "return_risk_final_model"
    )

    with mlflow.start_run(
        run_name="final_logistic_regression"
    ):

        mlflow.log_params(
            params
        )

        mlflow.log_metrics(
            metrics
        )

        mlflow.sklearn.log_model(
            model,
            name="model"
        )

    print("\nFinal Model: Logistic Regression")
    print("Parameters:")
    print(params)

    print("\nTest Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()