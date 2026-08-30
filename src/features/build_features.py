import pandas as pd
import pathlib
import yaml
import sys

def create_features(data):
    data=data.copy()

    data['order_date']=pd.to_datetime(data['order_date'])

    data['order_dayofweek']=data['order_date'].dt.dayofweek

    data['is_weekend']=(
        data['order_dayofweek'] >=5
    ).astype(int)

    data['order_month']=data['order_date'].dt.month

    data=data.sort_values(
        ["customer_id", "order_date"]
    )

    data["time_since_last_order"] = (
        data.groupby("customer_id")["order_date"]
        .diff()
        .dt.days
    )

    data["discount_amount"] = (
        data["order_value"]
        * data["discount_pct"]
        / 100
    )


    data["net_order_value"] = (
        data["order_value"]
        * (1 - data["discount_pct"] / 100)
    )

    data["delivery_delay_ratio"] = (
    data["delivery_days"]
    / data["expected_delivery_days"]
    )

    data = data.drop(
        columns=[
            "order_id",
            "customer_id",
            "order_date"
        ]
    )

    return data

def save_data(data,output_path):

    output_path = pathlib.Path(output_path)

    output_path.mkdir(
            parents=True,
            exist_ok=True
        )

    data.to_csv(
        output_path / "data_feature_engg.csv",index=False
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

    input_file = sys.argv[1]

    data_path = (
        home_dir / input_file
    )

    output_path = (
        home_dir / "data" / "processed"
    )

    data=pd.read_csv(data_path)

    data=create_features(
        data
    )

    save_data(data,output_path)

if __name__ == "__main__":
    main()
