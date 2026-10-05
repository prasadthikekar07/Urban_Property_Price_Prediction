import pandas as pd
import numpy as np


# =========================================================
# URBAN PROPERTY PRICE FORECASTING
# TIME-BASED SYNTHETIC DATASET GENERATOR
# =========================================================

np.random.seed(42)

# Number of records
n = 1200


# =========================================================
# CITY INFORMATION
# =========================================================

cities = {
    "Pune": 6500,
    "Mumbai": 15000,
    "Bengaluru": 9000,
    "Hyderabad": 6000,
    "Delhi": 8500,
    "Chennai": 6000,
    "Ahmedabad": 5000
}


# =========================================================
# CREATE BASIC DATA
# =========================================================

city_list = np.random.choice(
    list(cities.keys()),
    size=n
)

years = np.random.randint(
    2020,
    2026,
    size=n
)

months = np.random.randint(
    1,
    13,
    size=n
)


data = pd.DataFrame({
    "City": city_list,
    "Year": years,
    "Month": months,

    "Area_sqft": np.random.randint(
        500,
        2501,
        n
    ),

    "Bedrooms": np.random.randint(
        1,
        5,
        n
    ),

    "Metro_Distance_km": np.round(
        np.random.uniform(
            0.5,
            10.0,
            n
        ),
        2
    ),

    "Highway_Distance_km": np.round(
        np.random.uniform(
            1.0,
            12.0,
            n
        ),
        2
    ),

    "School_Distance_km": np.round(
        np.random.uniform(
            0.2,
            6.0,
            n
        ),
        2
    ),

    "Hospital_Distance_km": np.round(
        np.random.uniform(
            0.2,
            6.0,
            n
        ),
        2
    )
})


# =========================================================
# INFRASTRUCTURE DEVELOPMENT SIGNALS
# =========================================================

data["Road_Development"] = np.random.randint(
    0,
    2,
    n
)

data["Metro_Development"] = np.random.randint(
    0,
    2,
    n
)

data["Commercial_Development"] = np.random.randint(
    0,
    2,
    n
)


# =========================================================
# INFRASTRUCTURE DEVELOPMENT YEAR
# =========================================================

data["Infrastructure_Development_Year"] = np.random.randint(
    2021,
    2027,
    n
)


# =========================================================
# INFRASTRUCTURE DEVELOPMENT LEVEL
# =========================================================

data["Infrastructure_Level"] = (
    data["Road_Development"]
    + data["Metro_Development"]
    + data["Commercial_Development"]
)


# =========================================================
# CURRENT PROPERTY PRICE
# =========================================================

city_base_price = data["City"].map(cities)

# Time-related market growth
time_growth = (
    (data["Year"] - 2020) * 0.035
    + (data["Month"] - 1) * 0.001
)

# Location advantage
metro_advantage = (
    1
    - data["Metro_Distance_km"] * 0.012
)

highway_advantage = (
    1
    - data["Highway_Distance_km"] * 0.004
)

school_advantage = (
    1
    - data["School_Distance_km"] * 0.002
)

hospital_advantage = (
    1
    - data["Hospital_Distance_km"] * 0.002
)

# Calculate current price
data["Current_Price"] = (
    data["Area_sqft"]
    * city_base_price
    * metro_advantage
    * highway_advantage
    * school_advantage
    * hospital_advantage
    * (1 + time_growth)
    + data["Bedrooms"] * 350000
    + np.random.normal(
        0,
        350000,
        n
    )
)


# Prevent unrealistic prices
data["Current_Price"] = data[
    "Current_Price"
].clip(
    lower=1500000
)


# =========================================================
# FUTURE INFRASTRUCTURE IMPACT
# =========================================================

road_effect = (
    data["Road_Development"] * 0.035
)

metro_effect = (
    data["Metro_Development"] * 0.075
)

commercial_effect = (
    data["Commercial_Development"] * 0.045
)

# General annual market growth
market_growth = 0.04


# =========================================================
# FUTURE PRICE AFTER 12 MONTHS
# =========================================================

future_growth = (
    market_growth
    + road_effect
    + metro_effect
    + commercial_effect
)

data["Future_Price_12M"] = (
    data["Current_Price"]
    * (1 + future_growth)
    + np.random.normal(
        0,
        300000,
        n
    )
)


# Prevent negative values
data["Future_Price_12M"] = data[
    "Future_Price_12M"
].clip(
    lower=1500000
)


# =========================================================
# PRICE GROWTH
# =========================================================

data["Price_Growth_Percent"] = (
    (
        data["Future_Price_12M"]
        - data["Current_Price"]
    )
    / data["Current_Price"]
) * 100


# =========================================================
# PRICE TREND
# =========================================================

def determine_trend(growth):
    if growth >= 5:
        return "Strongly Increasing"
    elif growth >= 1:
        return "Increasing"
    elif growth > -1:
        return "Stable"
    else:
        return "Decreasing"


data["Price_Trend"] = data[
    "Price_Growth_Percent"
].apply(determine_trend)


# =========================================================
# ROUND NUMERIC VALUES
# =========================================================

data["Current_Price"] = data[
    "Current_Price"
].round(0)

data["Future_Price_12M"] = data[
    "Future_Price_12M"
].round(0)

data["Price_Growth_Percent"] = data[
    "Price_Growth_Percent"
].round(2)


# =========================================================
# SAVE DATASET
# =========================================================

data.to_csv(
    "data/property_data.csv",
    index=False
)


# =========================================================
# DISPLAY INFORMATION
# =========================================================

print("\n==============================================")
print("TIME-BASED PROPERTY DATASET GENERATED")
print("==============================================")

print("\nNumber of records:", len(data))

print("\nCities:")
print(data["City"].unique())

print("\nYears:")
print(sorted(data["Year"].unique()))

print("\nColumns:")
for column in data.columns:
    print("-", column)

print("\nFirst 5 records:")
print(data.head())

print("\nDataset saved successfully to:")
print("data/property_data.csv")

print("\nNOTE:")
print(
    "This is synthetic demonstration data and must "
    "not be treated as real property-market data."
)

print("\n==============================================")