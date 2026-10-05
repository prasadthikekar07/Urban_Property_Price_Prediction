import pandas as pd
import joblib


# =========================================================
# URBAN PROPERTY PRICE TREND FORECASTING
# PREDICTION PROGRAM
# =========================================================

# Load trained model
model = joblib.load("models/property_price_model.pkl")

print("\n==============================================")
print("   URBAN PROPERTY PRICE TREND FORECASTING")
print("==============================================")


# =========================================================
# PROPERTY DETAILS
# =========================================================

print("\n========== PROPERTY DETAILS ==========")

city = input(
    "Enter city (Pune/Mumbai/Bengaluru/Hyderabad/Delhi/Chennai/Ahmedabad): "
)

year = int(
    input("Enter year: ")
)

month = int(
    input("Enter month (1-12): ")
)

area = float(
    input("Enter property area (sqft): ")
)

bedrooms = int(
    input("Enter number of bedrooms: ")
)

metro_distance = float(
    input("Distance from metro (km): ")
)

highway_distance = float(
    input("Distance from highway (km): ")
)

school_distance = float(
    input("Distance from school (km): ")
)

hospital_distance = float(
    input("Distance from hospital (km): ")
)


# =========================================================
# INFRASTRUCTURE SIGNALS
# =========================================================

print("\n========== INFRASTRUCTURE DEVELOPMENT ==========")

road_development = int(
    input("Road development planned? (1=Yes, 0=No): ")
)

metro_development = int(
    input("Metro development planned? (1=Yes, 0=No): ")
)

commercial_development = int(
    input("Commercial development planned? (1=Yes, 0=No): ")
)

infrastructure_year = int(
    input("Infrastructure development year: ")
)


# Calculate infrastructure level

infrastructure_level = (
    road_development
    + metro_development
    + commercial_development
)


# =========================================================
# CURRENT PROPERTY PRICE
# =========================================================

current_price = float(
    input("\nEnter current property price (₹): ")
)


# =========================================================
# CREATE INPUT DATA
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
    "Road_Development": road_development,
    "Metro_Development": metro_development,
    "Commercial_Development": commercial_development,
    "Infrastructure_Development_Year": infrastructure_year,
    "Infrastructure_Level": infrastructure_level,
    "Current_Price": current_price
}])


# =========================================================
# PREDICT FUTURE PRICE
# =========================================================

future_price = model.predict(
    input_data
)[0]


# =========================================================
# PRICE GROWTH
# =========================================================

price_change = (
    future_price - current_price
)

if current_price > 0:

    growth_percentage = (
        price_change / current_price
    ) * 100

else:

    growth_percentage = 0


# =========================================================
# TREND CLASSIFICATION
# =========================================================

if growth_percentage >= 5:

    trend = "Strongly Increasing"

elif growth_percentage >= 1:

    trend = "Increasing"

elif growth_percentage > -1:

    trend = "Stable"

else:

    trend = "Decreasing"


# =========================================================
# INFRASTRUCTURE IMPACT
# =========================================================

if infrastructure_level == 3:

    infrastructure_impact = "High"

elif infrastructure_level == 2:

    infrastructure_impact = "Moderate"

elif infrastructure_level == 1:

    infrastructure_impact = "Low"

else:

    infrastructure_impact = "Minimal"


# =========================================================
# DISPLAY RESULT
# =========================================================

print("\n==============================================")
print("              FORECAST RESULT")
print("==============================================")

print(
    f"\nCity                    : {city}"
)

print(
    f"Forecast Period         : {month}/{year}"
)

print(
    f"Current Property Price  : ₹{current_price:,.2f}"
)

print(
    f"12-Month Forecast Price : ₹{future_price:,.2f}"
)

print(
    f"Expected Price Change   : ₹{price_change:,.2f}"
)

print(
    f"Expected Growth         : {growth_percentage:.2f}%"
)

print(
    f"Price Trend             : {trend}"
)

print(
    f"Infrastructure Impact   : {infrastructure_impact}"
)


# =========================================================
# INFRASTRUCTURE SUMMARY
# =========================================================

print("\n========== INFRASTRUCTURE SIGNALS ==========")

print(
    "Road Development       :",
    "Yes" if road_development else "No"
)

print(
    "Metro Development      :",
    "Yes" if metro_development else "No"
)

print(
    "Commercial Development :",
    "Yes" if commercial_development else "No"
)

print(
    "Infrastructure Year    :",
    infrastructure_year
)

print(
    "Infrastructure Level   :",
    infrastructure_level,
    "/ 3"
)


print("\n==============================================")
print("          FORECASTING COMPLETED")
print("==============================================")