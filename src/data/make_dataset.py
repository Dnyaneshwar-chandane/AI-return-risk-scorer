import pathlib
import yaml
import sys
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def load_data(data_path):
    df = pd.read_csv(data_path)
    return df


def split_data(df, test_split, seed):

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_split,
        random_state=seed
    )

    for train_idx, test_idx in splitter.split(
        df,
        df["returned"]
    ):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]

    return train, test


def save_data(train, test, output_path):

    output_path = pathlib.Path(output_path)

    output_path.mkdir(
        parents=True,
        exist_ok=True
    )

    train.to_csv(
        output_path / "train.csv",
        index=False
    )

    test.to_csv(
        output_path / "test.csv",
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
        params = yaml.safe_load(f)["make_dataset"]

    input_file = sys.argv[1]

    data_path = (
        home_dir / input_file
    )

    output_path = (
        home_dir / "data" / "processed"
    )

    data = load_data(
        data_path
    )

    train_data, test_data = split_data(
        data,
        params["test_split"],
        params["seed"]
    )

    save_data(
        train_data,
        test_data,
        output_path
    )


if __name__ == "__main__":
    main()