import pathlib
import sys
import pandas as pd


def feature_selection(train_data, test_data):

    drop_cols = [
        "net_order_value"
    ]

    train_data = train_data.drop(
        columns=drop_cols
    )

    test_data = test_data.drop(
        columns=drop_cols
    )

    return train_data, test_data


def save_data(train_data, test_data, output_path):

    output_path = pathlib.Path(output_path)

    output_path.mkdir(
        parents=True,
        exist_ok=True
    )

    train_data.to_csv(
        output_path / "selected_train.csv",
        index=False
    )

    test_data.to_csv(
        output_path / "selected_test.csv",
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

    train_file = sys.argv[1]
    test_file = sys.argv[2]

    train_path = home_dir / train_file
    test_path = home_dir / test_file

    output_path = (
        home_dir
        / "data"
        / "processed"
    )

    train_data = pd.read_csv(
        train_path
    )

    test_data = pd.read_csv(
        test_path
    )

    train_data, test_data = feature_selection(
        train_data,
        test_data
    )

    save_data(
        train_data,
        test_data,
        output_path
    )


if __name__ == "__main__":
    main()