import pathlib
import yaml
import sys
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

def preprocessing(X_train,X_test):

    numeric_cols=X_train.select_dtypes(include=['int64','float64']).columns.tolist()
    binary_cols = [col for col in numeric_cols if X_train[col].nunique() == 2]
    Numeric_cols = [col for col in numeric_cols if col not in binary_cols]
    categorical_cols=X_train.select_dtypes(include=['object']).columns.tolist()

    preprocessing=ColumnTransformer(
        [
            ('categorical',
              Pipeline(
                  [
                      ('impute',SimpleImputer(strategy='most_frequent')),
                      ('enode',OneHotEncoder(drop='first',handle_unknown='ignore'))
                  ]
              ),
              categorical_cols

             ),



             ('Numarical',
              Pipeline(
                  [
                      ('impute',SimpleImputer(strategy='median')),
                      ('Scaling',RobustScaler())
                  ]
              ),
              Numeric_cols
                 
             ),

             ('binary',
              Pipeline(
                  [
                      ('impute',SimpleImputer(strategy='most_frequent'))
                  ]
              ),
              binary_cols
                 
             )
        ],
        remainder='passthrough'

    )
    preprocessing.set_output(transform="pandas")
    X_train=preprocessing.fit_transform(X_train)
    X_test=preprocessing.transform(X_test)

    return X_train,X_test

def save_data(X_train, X_test, output_path):

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



def main():
    
    curr_dir = pathlib.Path(__file__)

    home_dir = (
        curr_dir
        .parent
        .parent
        .parent
    )

    params_file = (
        home_dir / "params.yaml"
    )

    with open(params_file) as f:
        params = yaml.safe_load(f)[
            "preprocessing"
        ]

    input_file = sys.argv[1]

    data_path = (
        home_dir / input_file
    )

    output_path = (
        home_dir / "data" / "processed"
    )

    TARGET = 'returned'
    train_features = pd.read_csv(data_path / '/train.csv')
    test_features = pd.read_csv(data_path / '/test.csv')
    X_train = train_features.drop(TARGET, axis=1)
    X_test = test_features.drop(TARGET,axis=1)

    X_train, X_test = preprocessing(
        X_train,X_test
    )

    save_data(
        X_train,X_test,output_path
    )
    


