import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Urban Property Price Trend Forecasting",
    page_icon="🏙️",
    layout="wide"
)


# =========================================================
# LOAD FILES
# =========================================================

@st.cache_resource
def load_production_model():
    return joblib.load(
        "models/property_price_model.pkl"
    )


@st.cache_resource
def load_evaluation_model():
    return joblib.load(
        "models/evaluation_model.pkl"
    )


@st.cache_data
def load_data():
    return pd.read_csv(
        "data/property_data.csv"
    )


@st.cache_resource
def load_metadata():
    return joblib.load(
        "models/model_metadata.pkl"
    )


production_model = load_production_model()
evaluation_model = load_evaluation_model()
data = load_data()
metadata = load_metadata()


# =========================================================
# TITLE
# =========================================================

st.title(
    "🏙️ Urban Property Price Trend Forecasting"
)

st.write(
    "Machine learning system for forecasting 12-month urban "
    "property prices using location, time and infrastructure "
    "development signals."
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("📍 Location & Time")

cities = sorted(
    data["City"].unique().tolist()
)

city = st.sidebar.selectbox(
    "City",
    cities
)

year = st.sidebar.number_input(
    "Reference Year",
    min_value=2020,
    max_value=2030,
    value=2025,
    step=1
)

month = st.sidebar.selectbox(
    "Reference Month",
    list(range(1, 13)),
    index=9
)


# =========================================================
# PROPERTY DETAILS
# =========================================================

st.sidebar.header("🏠 Property Details")

area = st.sidebar.number_input(
    "Property Area (sqft)",
    min_value=100,
    max_value=10000,
    value=1200,
    step=50
)

bedrooms = st.sidebar.number_input(
    "Number of Bedrooms",
    min_value=1,
    max_value=10,
    value=3
)

metro_distance = st.sidebar.number_input(
    "Distance from Metro (km)",
    min_value=0.0,
    max_value=50.0,
    value=2.0,
    step=0.1
)

highway_distance = st.sidebar.number_input(
    "Distance from Highway (km)",
    min_value=0.0,
    max_value=50.0,
    value=4.0,
    step=0.1
)

school_distance = st.sidebar.number_input(
    "Distance from School (km)",
    min_value=0.0,
    max_value=20.0,
    value=1.0,
    step=0.1
)

hospital_distance = st.sidebar.number_input(
    "Distance from Hospital (km)",
    min_value=0.0,
    max_value=20.0,
    value=1.5,
    step=0.1
)


# =========================================================
# INFRASTRUCTURE
# =========================================================

st.sidebar.header("🚧 Infrastructure Development")

road_development = st.sidebar.selectbox(
    "Road Development",
    ["No", "Yes"]
)

metro_development = st.sidebar.selectbox(
    "Metro Development",
    ["No", "Yes"]
)

commercial_development = st.sidebar.selectbox(
    "Commercial Development",
    ["No", "Yes"]
)

infrastructure_year = st.sidebar.number_input(
    "Infrastructure Development Year",
    min_value=2021,
    max_value=2030,
    value=2026,
    step=1
)

road_value = int(
    road_development == "Yes"
)

metro_value = int(
    metro_development == "Yes"
)

commercial_value = int(
    commercial_development == "Yes"
)

infrastructure_level = (
    road_value
    + metro_value
    + commercial_value
)


# =========================================================
# CURRENT PRICE
# =========================================================

st.sidebar.header("💰 Current Price")

current_price = st.sidebar.number_input(
    "Current Property Price (₹)",
    min_value=100000,
    max_value=100000000,
    value=8000000,
    step=100000
)


# =========================================================
# INPUT DATA
# =========================================================

input_data = pd.DataFrame([{
    "City": city,
    "Year": year,
    "Month": month,
    "Area_sqft": area,
    "Bedrooms": bedrooms,
    "Metro_Distance_km": metro_distance,
    "Highway_Distance_km": highway_distance,
    "School_Distance_km": school_distance,
    "Hospital_Distance_km": hospital_distance,
    "Road_Development": road_value,
    "Metro_Development": metro_value,
    "Commercial_Development": commercial_value,
    "Infrastructure_Development_Year": infrastructure_year,
    "Infrastructure_Level": infrastructure_level,
    "Current_Price": current_price
}])


# =========================================================
# FORECAST
# =========================================================

predict_clicked = st.sidebar.button(
    "🔮 Forecast Property Price",
    type="primary"
)


if predict_clicked:

    future_price = production_model.predict(
        input_data
    )[0]

    price_change = (
        future_price - current_price
    )

    growth_percentage = (
        price_change / current_price
    ) * 100


    if growth_percentage >= 5:
        trend = "Strongly Increasing"
    elif growth_percentage >= 1:
        trend = "Increasing"
    elif growth_percentage > -1:
        trend = "Stable"
    else:
        trend = "Decreasing"


    if infrastructure_level == 3:
        infrastructure_impact = "High"
    elif infrastructure_level == 2:
        infrastructure_impact = "Moderate"
    elif infrastructure_level == 1:
        infrastructure_impact = "Low"
    else:
        infrastructure_impact = "Minimal"


    # =====================================================
    # FORECAST RESULTS
    # =====================================================

    st.subheader(
        "📊 12-Month Forecast"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Current Price",
            f"₹{current_price:,.0f}"
        )

    with c2:
        st.metric(
            "Forecast Price",
            f"₹{future_price:,.0f}"
        )

    with c3:
        st.metric(
            "Expected Growth",
            f"{growth_percentage:.2f}%"
        )

    with c4:
        st.metric(
            "Trend",
            trend
        )


    st.divider()


    # =====================================================
    # SUMMARY
    # =====================================================

    st.subheader(
        "📍 Forecast Summary"
    )

    s1, s2 = st.columns(2)

    with s1:

        st.write(f"**City:** {city}")

        st.write(
            f"**Reference Period:** {month}/{year}"
        )

        st.write(
            f"**Property Area:** {area:,.0f} sqft"
        )

        st.write(
            f"**Bedrooms:** {bedrooms}"
        )


    with s2:

        st.write(
            f"**Infrastructure Level:** "
            f"{infrastructure_level}/3"
        )

        st.write(
            f"**Infrastructure Impact:** "
            f"{infrastructure_impact}"
        )

        st.write(
            f"**Development Year:** "
            f"{infrastructure_year}"
        )

        st.write(
            f"**Expected Price Change:** "
            f"₹{price_change:,.0f}"
        )


    # =====================================================
    # INFRASTRUCTURE TABLE
    # =====================================================

    st.subheader(
        "🚧 Infrastructure Development Signals"
    )

    infrastructure_table = pd.DataFrame({
        "Signal": [
            "Road Development",
            "Metro Development",
            "Commercial Development"
        ],
        "Status": [
            road_development,
            metro_development,
            commercial_development
        ]
    })

    st.dataframe(
        infrastructure_table,
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # PRICE CHART
    # =====================================================

    st.subheader(
        "💰 Current vs Forecast Price"
    )

    price_chart = pd.DataFrame(
        {
            "Price": [
                current_price,
                future_price
            ]
        },
        index=[
            "Current Price",
            "12-Month Forecast"
        ]
    )

    st.bar_chart(
        price_chart
    )


    # =====================================================
    # DOWNLOAD REPORT
    # =====================================================

    report = pd.DataFrame([{
        "City": city,
        "Reference_Year": year,
        "Reference_Month": month,
        "Area_sqft": area,
        "Bedrooms": bedrooms,
        "Metro_Distance_km": metro_distance,
        "Highway_Distance_km": highway_distance,
        "School_Distance_km": school_distance,
        "Hospital_Distance_km": hospital_distance,
        "Road_Development": road_development,
        "Metro_Development": metro_development,
        "Commercial_Development": commercial_development,
        "Infrastructure_Development_Year":
            infrastructure_year,
        "Infrastructure_Level":
            infrastructure_level,
        "Current_Price":
            round(current_price, 2),
        "Forecast_Price_12M":
            round(future_price, 2),
        "Expected_Price_Change":
            round(price_change, 2),
        "Expected_Growth_Percent":
            round(growth_percentage, 2),
        "Price_Trend": trend,
        "Infrastructure_Impact":
            infrastructure_impact
    }])

    st.download_button(
        label="📥 Download Forecast Report",
        data=report.to_csv(index=False),
        file_name="property_forecast_report.csv",
        mime="text/csv"
    )


# =========================================================
# INFRASTRUCTURE SCENARIOS
# =========================================================

st.divider()

st.subheader(
    "🔍 Infrastructure Scenario Analysis"
)

scenario_no = input_data.copy()

scenario_no["Road_Development"] = 0
scenario_no["Metro_Development"] = 0
scenario_no["Commercial_Development"] = 0
scenario_no["Infrastructure_Level"] = 0


scenario_selected = input_data.copy()


scenario_all = input_data.copy()

scenario_all["Road_Development"] = 1
scenario_all["Metro_Development"] = 1
scenario_all["Commercial_Development"] = 1
scenario_all["Infrastructure_Level"] = 3


price_no = production_model.predict(
    scenario_no
)[0]

price_selected = production_model.predict(
    scenario_selected
)[0]

price_all = production_model.predict(
    scenario_all
)[0]


scenario_table = pd.DataFrame({
    "Scenario": [
        "No Infrastructure",
        "Selected Infrastructure",
        "All Infrastructure"
    ],
    "Forecast Price (₹)": [
        round(price_no, 0),
        round(price_selected, 0),
        round(price_all, 0)
    ]
})

st.dataframe(
    scenario_table,
    use_container_width=True,
    hide_index=True
)

st.bar_chart(
    scenario_table.set_index("Scenario")
)


# =========================================================
# TIME-BASED MODEL EVALUATION
# =========================================================

st.divider()

st.subheader(
    "🤖 Time-Based Model Evaluation"
)


test_data = data[
    data["Year"] == 2025
].copy()


features = metadata["features"]


X_test = test_data[
    features
]

y_test = test_data[
    "Future_Price_12M"
]


test_predictions = evaluation_model.predict(
    X_test
)


mae = mean_absolute_error(
    y_test,
    test_predictions
)

rmse = mean_squared_error(
    y_test,
    test_predictions
) ** 0.5

r2 = r2_score(
    y_test,
    test_predictions
)


m1, m2, m3 = st.columns(3)

with m1:
    st.metric(
        "MAE",
        f"₹{mae:,.0f}"
    )

with m2:
    st.metric(
        "RMSE",
        f"₹{rmse:,.0f}"
    )

with m3:
    st.metric(
        "R² Score",
        f"{r2:.4f}"
    )


st.caption(
    "Evaluation uses the 2025 holdout period. "
    "The production model used for forecasts is trained "
    "on all available records."
)


# =========================================================
# ACTUAL VS PREDICTED
# =========================================================

st.subheader(
    "📉 Actual vs Predicted — 2025 Test Data"
)

comparison = test_data[
    [
        "City",
        "Year",
        "Month",
        "Current_Price",
        "Future_Price_12M"
    ]
].copy()

comparison[
    "Predicted_Future_Price"
] = test_predictions.round(0)

comparison[
    "Absolute_Error"
] = (
    comparison[
        "Future_Price_12M"
    ]
    - comparison[
        "Predicted_Future_Price"
    ]
).abs().round(0)


st.dataframe(
    comparison.head(25),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# ACTUAL VS PREDICTED GRAPH
# =========================================================

fig1, ax1 = plt.subplots(
    figsize=(10, 6)
)

ax1.scatter(
    y_test,
    test_predictions,
    alpha=0.6
)

minimum = min(
    y_test.min(),
    test_predictions.min()
)

maximum = max(
    y_test.max(),
    test_predictions.max()
)

ax1.plot(
    [minimum, maximum],
    [minimum, maximum]
)

ax1.set_xlabel(
    "Actual Future Price (₹)"
)

ax1.set_ylabel(
    "Predicted Future Price (₹)"
)

ax1.set_title(
    "Actual vs Predicted Future Property Prices"
)

ax1.grid(True)

plt.tight_layout()

st.pyplot(
    fig1
)


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

st.subheader(
    "📈 Feature Importance"
)

rf_model = production_model.named_steps[
    "model"
]

production_preprocessor = (
    production_model
    .named_steps["preprocessor"]
)

feature_names = (
    production_preprocessor
    .get_feature_names_out()
)

importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=True
)


fig2, ax2 = plt.subplots(
    figsize=(10, 8)
)

ax2.barh(
    importance["Feature"],
    importance["Importance"]
)

ax2.set_xlabel(
    "Importance"
)

ax2.set_ylabel(
    "Feature"
)

ax2.set_title(
    "Production Model Feature Importance"
)

plt.tight_layout()

st.pyplot(
    fig2
)


# =========================================================
# PRICE TREND DISTRIBUTION
# =========================================================

st.subheader(
    "📊 Price Trend Distribution"
)

trend_distribution = (
    data["Price_Trend"]
    .value_counts()
)

st.bar_chart(
    trend_distribution
)


# =========================================================
# CITY-WISE FORECAST
# =========================================================

st.subheader(
    "🏙️ City-wise Average 12-Month Forecast"
)

city_forecast = (
    data.groupby("City")[
        "Future_Price_12M"
    ]
    .mean()
    .round(0)
    .sort_values(ascending=False)
)

st.bar_chart(
    city_forecast
)


# =========================================================
# MODEL AUDIT
# =========================================================

st.divider()

st.subheader(
    "🔎 Model Audit"
)

audit1, audit2 = st.columns(2)

with audit1:

    st.write(
        "**Algorithm:** "
        + metadata["algorithm"]
    )

    st.write(
        "**Forecast Horizon:** "
        + metadata["forecast_horizon"]
    )

    st.write(
        "**Target Variable:** "
        + metadata["target"]
    )

    st.write(
        "**Dataset Records:** "
        + str(metadata["records"])
    )


with audit2:

    st.write(
        "**Evaluation Period:** "
        + metadata["evaluation_period"]
    )

    st.write(
        "**Evaluation MAE:** "
        f"₹{metadata['mae']:,.0f}"
    )

    st.write(
        "**Evaluation RMSE:** "
        f"₹{metadata['rmse']:,.0f}"
    )

    st.write(
        "**Evaluation R²:** "
        f"{metadata['r2_score']:.4f}"
    )


# =========================================================
# DATASET INFORMATION
# =========================================================

st.divider()

st.subheader(
    "📁 Dataset Information"
)

d1, d2, d3 = st.columns(3)

with d1:
    st.metric(
        "Records",
        len(data)
    )

with d2:
    st.metric(
        "Cities",
        data["City"].nunique()
    )

with d3:
    st.metric(
        "Years",
        data["Year"].nunique()
    )


with st.expander(
    "View Complete Dataset"
):

    st.dataframe(
        data,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# DATA SOURCE NOTE
# =========================================================

st.warning(
    "Current training data is synthetic demonstration data "
    "generated for software development and testing. "
    "The forecasts and model metrics must not be interpreted "
    "as real-world property-market valuations."
)


# =========================================================
# PROJECT INFORMATION
# =========================================================

st.divider()

st.subheader(
    "📌 Project Information"
)

p1, p2 = st.columns(2)

with p1:

    st.write(
        "**Machine Learning:** Random Forest Regression"
    )

    st.write(
        "**Location Features:** City"
    )

    st.write(
        "**Time Features:** Year and Month"
    )

with p2:

    st.write(
        "**Infrastructure Signals:** Road, Metro, Commercial"
    )

    st.write(
        "**Forecast Horizon:** 12 Months"
    )

    st.write(
        "**Evaluation Method:** Time-based holdout"
    )


# =========================================================
# FOOTER
# =========================================================

st.caption(
    "Urban Property Price Trend Forecasting using "
    "Infrastructure Development Signals"
)