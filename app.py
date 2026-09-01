import streamlit as st
import pandas as pd
import joblib


# -------------------------
# Load artifacts
# -------------------------

preprocessor = joblib.load(
    "models/preprocessor.pkl"
)

model = joblib.load(
    "models/return_risk_model.pkl"
)


# -------------------------
# Page
# -------------------------

st.set_page_config(
    page_title="Return Risk Predictor",
    page_icon="📦",
    layout="centered"
)

st.title("📦 Customer Return Risk Predictor")

st.write(
    "Predict whether an order is likely to be returned."
)


# -------------------------
# Input form
# -------------------------

category = st.selectbox(
    "Category",
    ["Beauty", "Books", "Electronics", "Fashion"]
)

region = st.selectbox(
    "Region",
    ["North", "South", "East", "West"]
)

order_value = st.number_input(
    "Order Value",
    min_value=0.0,
    value=1000.0
)

payment_method = st.selectbox(
    "Payment Method",
    ["COD", "Card", "UPI", "Net Banking"]
)

is_cod = st.selectbox(
    "Is COD",
    [0, 1]
)

discount_pct = st.number_input(
    "Discount %",
    min_value=0.0,
    max_value=100.0,
    value=10.0
)

is_high_discount = st.selectbox(
    "Is High Discount",
    [0, 1]
)

expected_delivery_days = st.number_input(
    "Expected Delivery Days",
    min_value=0.0,
    value=5.0
)

delivery_days = st.number_input(
    "Actual Delivery Days",
    min_value=0.0,
    value=5.0
)

delivery_delay = st.number_input(
    "Delivery Delay",
    value=0.0
)

is_delivery_delayed = st.selectbox(
    "Is Delivery Delayed",
    [0, 1]
)

customer_previous_orders = st.number_input(
    "Previous Orders",
    min_value=0,
    value=5
)

customer_previous_returns = st.number_input(
    "Previous Returns",
    min_value=0,
    value=1
)

customer_return_rate = st.number_input(
    "Customer Return Rate",
    min_value=0.0,
    max_value=1.0,
    value=0.2
)

is_high_value = st.selectbox(
    "Is High Value",
    [0, 1]
)

order_dayofweek = st.number_input(
    "Order Day of Week",
    min_value=0,
    max_value=6,
    value=2
)

is_weekend = st.selectbox(
    "Is Weekend",
    [0, 1]
)

order_month = st.number_input(
    "Order Month",
    min_value=1,
    max_value=12,
    value=8
)

time_since_last_order = st.number_input(
    "Time Since Last Order",
    min_value=0.0,
    value=30.0
)

discount_amount = st.number_input(
    "Discount Amount",
    min_value=0.0,
    value=100.0
)

delivery_delay_ratio = st.number_input(
    "Delivery Delay Ratio",
    min_value=0.0,
    value=0.0
)


# -------------------------
# Create input dataframe
# -------------------------

input_data = pd.DataFrame({
    "category": [category],
    "region": [region],
    "order_value": [order_value],
    "payment_method": [payment_method],
    "is_cod": [is_cod],
    "discount_pct": [discount_pct],
    "is_high_discount": [is_high_discount],
    "expected_delivery_days": [expected_delivery_days],
    "delivery_days": [delivery_days],
    "delivery_delay": [delivery_delay],
    "is_delivery_delayed": [is_delivery_delayed],
    "customer_previous_orders": [customer_previous_orders],
    "customer_previous_returns": [customer_previous_returns],
    "customer_return_rate": [customer_return_rate],
    "is_high_value": [is_high_value],
    "order_dayofweek": [order_dayofweek],
    "is_weekend": [is_weekend],
    "order_month": [order_month],
    "time_since_last_order": [time_since_last_order],
    "discount_amount": [discount_amount],
    "delivery_delay_ratio": [delivery_delay_ratio]
})


# -------------------------
# Prediction
# -------------------------

if st.button(
    "Predict Return Risk",
    type="primary"
):

    X = preprocessor.transform(
        input_data
    )

    probability = model.predict_proba(X)[0][1]

    prediction = int(
        probability >= 0.5
    )

    st.divider()

    if prediction == 1:

        st.error(
            "⚠️ HIGH RETURN RISK"
        )

    else:

        st.success(
            "✅ LOW RETURN RISK"
        )

    st.metric(
        "Return Probability",
        f"{probability:.2%}"
    )