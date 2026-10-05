import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os


# ==========================================
# URBAN PROPERTY PRICE TREND FORECASTING
# DATA VISUALIZATION
# ==========================================

# Load dataset
data = pd.read_csv("data/property_data.csv")

# Create output folder for graphs
os.makedirs("graphs", exist_ok=True)


# ==========================================
# 1. CURRENT PRICE VS FUTURE PRICE
# ==========================================

plt.figure(figsize=(10, 6))

plt.scatter(
    data["Current_Price"],
    data["Future_Price_12M"],
    alpha=0.7
)

plt.xlabel("Current Property Price (₹)")
plt.ylabel("Future Price After 12 Months (₹)")
plt.title("Current Price vs Future Property Price")

plt.grid(True)
plt.tight_layout()

plt.savefig("graphs/current_vs_future_price.png")
plt.show()


# ==========================================
# 2. INFRASTRUCTURE IMPACT
# ==========================================

data["Infrastructure_Score"] = (
    data["Road_Development"]
    + data["Metro_Development"]
    + data["Commercial_Development"]
)

average_prices = data.groupby(
    "Infrastructure_Score"
)["Future_Price_12M"].mean()

plt.figure(figsize=(9, 6))

average_prices.plot(
    kind="bar",
    xlabel="Infrastructure Development Score",
    ylabel="Average Future Property Price (₹)",
    title="Infrastructure Development Impact on Future Prices"
)

plt.grid(axis="y")
plt.tight_layout()

plt.savefig("graphs/infrastructure_impact.png")
plt.show()


# ==========================================
# 3. PRICE DISTRIBUTION
# ==========================================

plt.figure(figsize=(10, 6))

plt.hist(
    data["Future_Price_12M"],
    bins=20,
    edgecolor="black"
)

plt.xlabel("Future Property Price (₹)")
plt.ylabel("Number of Properties")
plt.title("Distribution of Forecasted Property Prices")

plt.grid(axis="y")
plt.tight_layout()

plt.savefig("graphs/price_distribution.png")
plt.show()


# ==========================================
# 4. FEATURE IMPORTANCE
# ==========================================

model = joblib.load(
    "models/property_price_model.pkl"
)

features = [
    "Area_sqft",
    "Bedrooms",
    "Metro_Distance_km",
    "Highway_Distance_km",
    "School_Distance_km",
    "Hospital_Distance_km",
    "Road_Development",
    "Metro_Development",
    "Commercial_Development",
    "Current_Price"
]

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=True
)

plt.figure(figsize=(10, 7))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance in Property Price Forecasting")

plt.tight_layout()

plt.savefig("graphs/feature_importance.png")
plt.show()


print("\n==========================================")
print("VISUALIZATION COMPLETED SUCCESSFULLY")
print("==========================================")

print("\nGraphs saved in:")
print("graphs/")