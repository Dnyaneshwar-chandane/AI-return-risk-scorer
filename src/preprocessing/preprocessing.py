import pathlib
import yaml
import sys
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib

def preprocessing(X_train, X_test):

    numeric_cols = X_train.select_dtypes(
        include=['int64', 'float64']
    ).columns.tolist()

    binary_cols = [
        col for col in numeric_cols
        if X_train[col].nunique() == 2
    ]

    Numeric_cols = [
        col for col in numeric_cols
        if col not in binary_cols
    ]

    categorical_cols = X_train.select_dtypes(
        include=['object']
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        [
            (
                'categorical',
                Pipeline(
                    [
                        (
                            'impute',
                            SimpleImputer(
                                strategy='most_frequent'
                            )
                        ),
                        (
                            'encode',
                            OneHotEncoder(
                                drop='first',
                                handle_unknown='ignore',
                                sparse_output=False
                            )
                        )
                    ]
                ),
                categorical_cols
            ),

            (
                'Numerical',
                Pipeline(
                    [
                        (
                            'impute',
                            SimpleImputer(
                                strategy='median'
                            )
                        ),
                        (
                            'Scaling',
                            RobustScaler()
                        )
                    ]
                ),
                Numeric_cols
            ),

            (
                'binary',
                Pipeline(
                    [
                        (
                            'impute',
                            SimpleImputer(
                                strategy='most_frequent'
                            )
                        )
                    ]
                ),
                binary_cols
            )
        ],
        remainder='passthrough'
    )

    preprocessor.set_output(
        transform="pandas"
    )

    X_train = preprocessor.fit_transform(
        X_train
    )

    X_test = preprocessor.transform(
        X_test
    )

    return X_train, X_test, preprocessor

def save_data(X_train, X_test, preprocessor, output_path, home_dir):

    output_path = pathlib.Path(output_path)

    output_path.mkdir(
        parents=True,
        exist_ok=True
    )

    X_train.to_csv(
        output_path / "X_train_trf.csv",
        index=False
    )

    X_test.to_csv(
        output_path / "X_test_trf.csv",
        index=False
    )

    model_path = home_dir / "models"

    model_path.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        preprocessor,
        model_path / "preprocessor.pkl"
    )


def main():

    curr_dir = pathlib.Path(__file__)

    home_dir = (
        curr_dir
        .parent
        .parent
        .parent
    )

    input_file = sys.argv[1]

    data_path = (
        home_dir / input_file
    )

    output_path = (
        home_dir / "data" / "processed"
    )

    TARGET = "returned"

    train_features = pd.read_csv(
        data_path / "selected_train.csv"
    )

    test_features = pd.read_csv(
        data_path / "selected_test.csv"
    )

    X_train = train_features.drop(
        TARGET,
        axis=1
    )

    X_test = test_features.drop(
        TARGET,
        axis=1
    )

    X_train, X_test, preprocessor = preprocessing(
        X_train, X_test
            )

    save_data(
        X_train,
        X_test,
        preprocessor,
        output_path,
        home_dir
    )


if __name__ == "__main__":
    main()
