"""
Synthetic E-commerce Order Data Generator — Return-Risk Scorer V5
=================================================================

Purpose
-------
Generate realistic synthetic e-commerce order data for a return-risk
classification problem.

Design principles
-----------------
1. Customer history is generated sequentially.
2. Historical features contain ONLY information available before
   the current order.
3. Orders are generated chronologically.
4. Return probability is influenced by realistic business signals.
5. The final returned target is used to update future customer history.
6. Latent customer behaviour is NOT exposed as a model feature.
7. Data generation is reproducible using a random seed.
8. Basic data-quality validation is performed before saving.

Important
---------
This script ONLY generates and validates the RAW dataset.

Train/test splitting is intentionally NOT done here.

Train/test splitting will be handled later in the modeling pipeline.

Usage
-----
python src/data/generate_return_risk_data.py \
    --rows 10000 \
    --out data/raw/orders.csv

Custom thresholds
-----------------
python src/data/generate_return_risk_data.py \
    --rows 10000 \
    --out data/raw/orders.csv \
    --high-discount-threshold 40 \
    --delay-threshold 2
"""

import argparse
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# =====================================================================
# CONFIGURATION
# =====================================================================

CATEGORIES = [
    "Fashion",
    "Electronics",
    "Footwear",
    "Home & Kitchen",
    "Beauty",
    "Mobiles",
    "Toys",
    "Sports",
    "Books",
    "Furniture",
]


REGIONS = [
    "North",
    "South",
    "East",
    "West",
    "Central",
]


PAYMENT_METHODS = [
    "COD",
    "UPI",
    "Card",
    "NetBanking",
    "Wallet",
]


# Natural return tendency by category.
# Fashion and Footwear have higher return tendency
# because of sizing / fit-related issues.
CATEGORY_RETURN_BIAS = {
    "Fashion": 0.20,
    "Footwear": 0.18,
    "Electronics": 0.07,
    "Home & Kitchen": 0.06,
    "Beauty": 0.05,
    "Mobiles": 0.07,
    "Toys": 0.05,
    "Sports": 0.06,
    "Books": 0.025,
    "Furniture": 0.08,
}


# Approximate realistic order-value ranges by category.
CATEGORY_PRICE_RANGE = {
    "Fashion": (500, 5000),
    "Electronics": (1000, 25000),
    "Footwear": (700, 7000),
    "Home & Kitchen": (300, 10000),
    "Beauty": (200, 4000),
    "Mobiles": (5000, 60000),
    "Toys": (200, 5000),
    "Sports": (500, 10000),
    "Books": (150, 2000),
    "Furniture": (3000, 50000),
}


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def sigmoid(x):
    """
    Convert log-odds into probability.
    """
    return 1.0 / (1.0 + np.exp(-x))


def generate_customer_profiles(rng, n_customers):
    """
    Create customers with a hidden behavioural tendency.

    latent_return_tendency is used only to simulate realistic
    customer behaviour.

    It is NOT included in the final dataset.
    """

    customers = []

    for i in range(n_customers):

        customer_id = f"CUST{100000 + i}"

        latent_return_tendency = rng.beta(
            2,
            12,
        )

        customers.append(
            {
                "customer_id": customer_id,
                "latent_return_tendency": (
                    latent_return_tendency
                ),
                "previous_orders": 0,
                "previous_returns": 0,
            }
        )

    return customers


def generate_order_value(rng, category):
    """
    Generate category-aware order values.

    Log-normal distribution creates right-skewed
    monetary values, which is common in transaction data.
    """

    low, high = CATEGORY_PRICE_RANGE[category]

    midpoint = (low + high) / 2

    value = rng.lognormal(
        mean=np.log(midpoint),
        sigma=0.50,
    )

    value = np.clip(
        value,
        low,
        high,
    )

    return round(
        float(value),
        2,
    )


def generate_order_dates(
    rng,
    n_rows,
    days_back=180,
):
    """
    Generate chronological order dates.

    Chronological generation is important because customer
    history must only contain information from previous orders.
    """

    start_date = (
        datetime.today()
        - timedelta(days=days_back)
    )

    offsets = rng.integers(
        0,
        days_back + 1,
        size=n_rows,
    )

    offsets.sort()

    return [
        start_date
        + timedelta(days=int(offset))
        for offset in offsets
    ]


# =====================================================================
# DATA QUALITY VALIDATION
# =====================================================================

def validate_dataset(df):
    """
    Validate important data-quality and business constraints.
    """

    required_columns = [
        "order_id",
        "customer_id",
        "order_date",
        "category",
        "region",
        "order_value",
        "payment_method",
        "is_cod",
        "discount_pct",
        "is_high_discount",
        "expected_delivery_days",
        "delivery_days",
        "delivery_delay",
        "is_delivery_delayed",
        "customer_previous_orders",
        "customer_previous_returns",
        "customer_return_rate",
        "is_high_value",
        "returned",
    ]

    # ---------------------------------------------------------------
    # Required columns
    # ---------------------------------------------------------------

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # ---------------------------------------------------------------
    # Missing values
    # ---------------------------------------------------------------

    missing_values = (
        df[required_columns]
        .isnull()
        .sum()
    )

    missing_values = (
        missing_values[
            missing_values > 0
        ]
    )

    if not missing_values.empty:

        raise ValueError(
            "Missing values detected:\n"
            f"{missing_values}"
        )

    # ---------------------------------------------------------------
    # Target
    # ---------------------------------------------------------------

    if not set(
        df["returned"].unique()
    ).issubset({0, 1}):

        raise ValueError(
            "returned must contain only 0 and 1."
        )

    # ---------------------------------------------------------------
    # Customer history
    # ---------------------------------------------------------------

    if (
        df["customer_previous_returns"]
        > df["customer_previous_orders"]
    ).any():

        raise ValueError(
            "Previous returns cannot exceed previous orders."
        )

    if (
        (df["customer_return_rate"] < 0)
        | (df["customer_return_rate"] > 1)
    ).any():

        raise ValueError(
            "customer_return_rate must be between 0 and 1."
        )

    # ---------------------------------------------------------------
    # Delivery
    # ---------------------------------------------------------------

    expected_delay = (
        df["delivery_days"]
        - df["expected_delivery_days"]
    ).clip(lower=0)

    if not np.array_equal(
        df["delivery_delay"].values,
        expected_delay.values,
    ):

        raise ValueError(
            "delivery_delay calculation is inconsistent."
        )

    # ---------------------------------------------------------------
    # Payment
    # ---------------------------------------------------------------

    expected_cod = (
        df["payment_method"] == "COD"
    ).astype(int)

    if not np.array_equal(
        df["is_cod"].values,
        expected_cod.values,
    ):

        raise ValueError(
            "is_cod is inconsistent with payment_method."
        )

    # ---------------------------------------------------------------
    # Discount
    # ---------------------------------------------------------------

    if (
        (df["discount_pct"] < 0)
        | (df["discount_pct"] > 70)
    ).any():

        raise ValueError(
            "discount_pct is outside expected range."
        )

    # ---------------------------------------------------------------
    # Order IDs
    # ---------------------------------------------------------------

    if df["order_id"].duplicated().any():

        raise ValueError(
            "Duplicate order IDs found."
        )

    # ---------------------------------------------------------------
    # Date ordering
    # ---------------------------------------------------------------

    dates = pd.to_datetime(
        df["order_date"]
    )

    if not dates.is_monotonic_increasing:

        raise ValueError(
            "Orders are not chronologically ordered."
        )

    print(
        "\n✓ Dataset validation passed."
    )


# =====================================================================
# DATASET SUMMARY
# =====================================================================

def print_dataset_summary(df):

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SYNTHETIC RETURN-RISK DATASET"
    )

    print(
        "=" * 70
    )

    print(
        f"Rows          : {len(df):,}"
    )

    print(
        f"Columns       : {len(df.columns)}"
    )

    print(
        f"Return rate   : "
        f"{df['returned'].mean():.2%}"
    )

    print(
        f"Date range    : "
        f"{df['order_date'].min()} "
        f"→ "
        f"{df['order_date'].max()}"
    )

    # ---------------------------------------------------------------
    # Target distribution
    # ---------------------------------------------------------------

    print(
        "\nTarget distribution:"
    )

    target_distribution = (
        df["returned"]
        .value_counts(
            normalize=True
        )
        .sort_index()
    )

    target_distribution.index = [
        "Not Returned"
        if value == 0
        else "Returned"
        for value in target_distribution.index
    ]

    print(
        target_distribution
        .map(
            lambda value:
            f"{value:.2%}"
        )
        .to_string()
    )

    # ---------------------------------------------------------------
    # Payment method
    # ---------------------------------------------------------------

    print(
        "\nReturn rate by payment method:"
    )

    print(
        df.groupby(
            "payment_method"
        )["returned"]
        .mean()
        .sort_values(
            ascending=False
        )
        .map(
            lambda value:
            f"{value:.2%}"
        )
        .to_string()
    )

    # ---------------------------------------------------------------
    # Category
    # ---------------------------------------------------------------

    print(
        "\nReturn rate by category:"
    )

    print(
        df.groupby(
            "category"
        )["returned"]
        .mean()
        .sort_values(
            ascending=False
        )
        .map(
            lambda value:
            f"{value:.2%}"
        )
        .to_string()
    )

    # ---------------------------------------------------------------
    # Discount
    # ---------------------------------------------------------------

    print(
        "\nReturn rate by high-discount flag:"
    )

    print(
        df.groupby(
            "is_high_discount"
        )["returned"]
        .mean()
        .map(
            lambda value:
            f"{value:.2%}"
        )
        .to_string()
    )

    # ---------------------------------------------------------------
    # Delivery delay
    # ---------------------------------------------------------------

    print(
        "\nReturn rate by delivery-delay flag:"
    )

    print(
        df.groupby(
            "is_delivery_delayed"
        )["returned"]
        .mean()
        .map(
            lambda value:
            f"{value:.2%}"
        )
        .to_string()
    )


# =====================================================================
# MAIN GENERATOR
# =====================================================================

def generate_orders(
    n_rows=10000,
    seed=42,
    high_discount_threshold=40,
    delay_threshold=2,
    days_back=180,
):
    """
    Generate the raw synthetic order dataset.
    """

    if n_rows < 100:

        raise ValueError(
            "Generate at least 100 rows."
        )

    if days_back < 30:

        raise ValueError(
            "days_back should be at least 30."
        )

    rng = np.random.default_rng(
        seed
    )

    random.seed(seed)

    # Approximately one customer for every four orders.
    n_customers = max(
        500,
        n_rows // 4,
    )

    customers = (
        generate_customer_profiles(
            rng,
            n_customers,
        )
    )

    order_dates = (
        generate_order_dates(
            rng,
            n_rows,
            days_back,
        )
    )

    rows = []

    # ================================================================
    # Sequential generation
    # ================================================================

    for i in range(n_rows):

        # ------------------------------------------------------------
        # Select customer
        # ------------------------------------------------------------

        customer = random.choice(
            customers
        )

        customer_id = customer[
            "customer_id"
        ]

        previous_orders = customer[
            "previous_orders"
        ]

        previous_returns = customer[
            "previous_returns"
        ]

        latent_tendency = customer[
            "latent_return_tendency"
        ]

        # ------------------------------------------------------------
        # Order information
        # ------------------------------------------------------------

        order_id = (
            f"ORD{1000000 + i}"
        )

        order_date = order_dates[i]

        # ------------------------------------------------------------
        # Category
        # ------------------------------------------------------------

        category = random.choices(
            CATEGORIES,
            weights=[
                1.20,
                1.00,
                1.10,
                1.00,
                0.90,
                1.00,
                0.70,
                0.70,
                0.60,
                0.50,
            ],
            k=1,
        )[0]

        # ------------------------------------------------------------
        # Order value
        # ------------------------------------------------------------

        order_value = (
            generate_order_value(
                rng,
                category,
            )
        )

        # ------------------------------------------------------------
        # Payment method
        # ------------------------------------------------------------

        payment_method = random.choices(
            PAYMENT_METHODS,
            weights=[
                0.35,
                0.30,
                0.20,
                0.08,
                0.07,
            ],
            k=1,
        )[0]

        is_cod = int(
            payment_method == "COD"
        )

        # ------------------------------------------------------------
        # Discount
        # ------------------------------------------------------------

        discount_pct = round(
            float(
                np.clip(
                    rng.normal(
                        15,
                        12,
                    ),
                    0,
                    70,
                )
            ),
            1,
        )

        is_high_discount = int(
            discount_pct
            >= high_discount_threshold
        )

        # ------------------------------------------------------------
        # Delivery
        # ------------------------------------------------------------

        expected_delivery_days = int(
            rng.choice(
                [2, 3, 4, 5],
                p=[
                    0.10,
                    0.25,
                    0.45,
                    0.20,
                ],
            )
        )

        delivery_days = int(
            np.clip(
                rng.normal(
                    expected_delivery_days + 1,
                    2,
                ),
                1,
                15,
            )
        )

        delivery_delay = max(
            0,
            delivery_days
            - expected_delivery_days,
        )

        is_delivery_delayed = int(
            delivery_delay
            >= delay_threshold
        )

        # ------------------------------------------------------------
        # Region
        # ------------------------------------------------------------

        region = random.choice(
            REGIONS
        )

        # ------------------------------------------------------------
        # Customer history
        #
        # ONLY information from previous orders.
        # ------------------------------------------------------------

        if previous_orders > 0:

            customer_return_rate = (
                previous_returns
                / previous_orders
            )

        else:

            customer_return_rate = 0.0

        customer_return_rate = round(
            customer_return_rate,
            3,
        )

        # ------------------------------------------------------------
        # High-value flag
        # ------------------------------------------------------------

        category_low, category_high = (
            CATEGORY_PRICE_RANGE[
                category
            ]
        )

        category_midpoint = (
            category_low
            + category_high
        ) / 2

        is_high_value = int(
            order_value
            > category_midpoint
        )

        # ------------------------------------------------------------
        # Return probability
        # ------------------------------------------------------------

        base_rate = (
            CATEGORY_RETURN_BIAS[
                category
            ]
        )

        base_log_odds = np.log(
            base_rate
            / (1 - base_rate)
        )

        log_odds = base_log_odds

        # COD increases return risk.
        log_odds += (
            0.55 * is_cod
        )

        # Heavy discount increases impulse-buy risk.
        log_odds += (
            0.45
            * is_high_discount
        )

        # Delivery delays increase dissatisfaction.
        log_odds += (
            0.22
            * delivery_delay
        )

        # Previous customer return behaviour.
        log_odds += (
            1.80
            * customer_return_rate
        )

        # High-value orders have slightly higher risk.
        log_odds += (
            0.20
            * is_high_value
        )

        # Hidden customer behaviour.
        log_odds += (
            1.20
            * latent_tendency
        )

        # Unexplained randomness.
        log_odds += rng.normal(
            0,
            0.40,
        )

        # ------------------------------------------------------------
        # Convert to probability
        # ------------------------------------------------------------

        return_probability = sigmoid(
            log_odds
        )

        return_probability = float(
            np.clip(
                return_probability,
                0.01,
                0.90,
            )
        )

        # ------------------------------------------------------------
        # Generate target
        # ------------------------------------------------------------

        returned = int(
            rng.random()
            < return_probability
        )

        # ------------------------------------------------------------
        # Save row
        # ------------------------------------------------------------

        rows.append(
            {
                "order_id": order_id,
                "customer_id": customer_id,
                "order_date": (
                    order_date.strftime(
                        "%Y-%m-%d"
                    )
                ),

                "category": category,
                "region": region,

                "order_value": order_value,

                "payment_method": (
                    payment_method
                ),
                "is_cod": is_cod,

                "discount_pct": discount_pct,
                "is_high_discount": (
                    is_high_discount
                ),

                "expected_delivery_days": (
                    expected_delivery_days
                ),
                "delivery_days": delivery_days,
                "delivery_delay": (
                    delivery_delay
                ),
                "is_delivery_delayed": (
                    is_delivery_delayed
                ),

                "customer_previous_orders": (
                    previous_orders
                ),
                "customer_previous_returns": (
                    previous_returns
                ),
                "customer_return_rate": (
                    customer_return_rate
                ),

                "is_high_value": (
                    is_high_value
                ),

                "returned": returned,
            }
        )

        # ------------------------------------------------------------
        # Update customer history AFTER target generation.
        # ------------------------------------------------------------

        customer[
            "previous_orders"
        ] += 1

        customer[
            "previous_returns"
        ] += returned

    return pd.DataFrame(rows)


# =====================================================================
# MAIN
# =====================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic e-commerce "
            "return-risk data."
        )
    )

    parser.add_argument(
        "--rows",
        type=int,
        default=10000,
        help="Number of orders to generate.",
    )

    parser.add_argument(
        "--out",
        type=str,
        default="orders.csv",
        help="Output CSV path.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--high-discount-threshold",
        type=float,
        default=40,
        help=(
            "Discount percentage at or above "
            "which an order is high-discount."
        ),
    )

    parser.add_argument(
        "--delay-threshold",
        type=int,
        default=2,
        help=(
            "Delivery delay days at or above "
            "which an order is delayed."
        ),
    )

    parser.add_argument(
        "--days-back",
        type=int,
        default=180,
        help=(
            "Number of days over which orders "
            "are distributed."
        ),
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------
    # Generate
    # -----------------------------------------------------------------

    df = generate_orders(
        n_rows=args.rows,
        seed=args.seed,
        high_discount_threshold=(
            args.high_discount_threshold
        ),
        delay_threshold=(
            args.delay_threshold
        ),
        days_back=args.days_back,
    )

    # -----------------------------------------------------------------
    # Validate
    # -----------------------------------------------------------------

    validate_dataset(df)

    # -----------------------------------------------------------------
    # Save raw dataset
    # -----------------------------------------------------------------

    df.to_csv(
        args.out,
        index=False,
    )

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    print_dataset_summary(df)

    print("\n" + "=" * 70)
    print("FILES")
    print("=" * 70)

    print(
        f"Raw dataset : {args.out}"
    )

    print(
        "\n✓ Raw synthetic dataset generated successfully."
    )

    print(
        "✓ No train/test split performed."
    )


if __name__ == "__main__":
    main()