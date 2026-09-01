import pathlib
import sys
import yaml
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def evaluate_model(model, X_train, y_train, X_test, y_test):

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

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

    return model, metrics


def log_model(model, name):

    if name == "XGBoost":

        mlflow.xgboost.log_model(
            model,
            name="model"
        )

    elif name == "LightGBM":

        mlflow.lightgbm.log_model(
            model,
            name="model"
        )

    else:

        mlflow.sklearn.log_model(
            model,
            name="model"
        )


def main():

    curr_dir = pathlib.Path(__file__)

    home_dir = (
        curr_dir
        .parent
        .parent
        .parent
    )



    params_file = home_dir / "params.yaml"

    with open(params_file) as f:
        params = yaml.safe_load(f)["model"]



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



    models = {

        "Logistic Regression":
            LogisticRegression(
                **params["logistic_regression"],
                random_state=42
            ),

        "Random Forest":
            RandomForestClassifier(
                **params["random_forest"],
                random_state=42
            ),

        "XGBoost":
            XGBClassifier(
                **params["xgboost"],
                random_state=42,
                eval_metric="logloss"
            ),

        "LightGBM":
            LGBMClassifier(
                **params["lightgbm"],
                random_state=42,
                verbosity=-1
            ),

        "Extra Trees":
            ExtraTreesClassifier(
                **params["extra_trees"],
                random_state=42
            ),

        "Hist Gradient Boosting":
            HistGradientBoostingClassifier(
                **params["hist_gradient_boosting"],
                random_state=42
            ),

        "SVM":
            SVC(
                **params["svm"],
                probability=True,
                random_state=42
            ),
        "KNN":
            KNeighborsClassifier(
                **params["knn"]
            )
    }


    mlflow.set_experiment(
        "return_risk_model_comparison"
    )

    for name, model in models.items():

        with mlflow.start_run(
            run_name=name
        ):

            trained_model, metrics = evaluate_model(
                model,
                X_train,
                y_train,
                X_test,
                y_test
            )

            mlflow.log_params(
                model.get_params()
            )

            mlflow.log_metrics(
                metrics
            )

        
            log_model(
                trained_model,
                name
            )

            print(
                f"{name}: "
                f"F1={metrics['f1_class_1']:.4f}, "
                f"Recall={metrics['recall_class_1']:.4f}, "
                f"PR-AUC={metrics['pr_auc']:.4f}"
            )


if __name__ == "__main__":
    main()