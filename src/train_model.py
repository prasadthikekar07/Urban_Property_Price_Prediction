import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# URBAN PROPERTY PRICE TREND FORECASTING
# PRODUCTION MODEL + TIME-BASED EVALUATION
# =========================================================

DATA_PATH = "data/property_data.csv"
MODEL_PATH = "models/property_price_model.pkl"
EVALUATION_MODEL_PATH = "models/evaluation_model.pkl"


print("\n==============================================")
print(" URBAN PROPERTY PRICE TREND FORECASTING")
print("==============================================")


# =========================================================
# LOAD DATA
# =========================================================

data = pd.read_csv(DATA_PATH)

data = data.sort_values(
    ["Year", "Month"]
).reset_index(drop=True)

print("\nDataset records:", len(data))
print(
    "Year range:",
    data["Year"].min(),
    "-",
    data["Year"].max()
)


# =========================================================
# FEATURES
# =========================================================

features = [
    "City",
    "Year",
    "Month",
    "Area_sqft",
    "Bedrooms",
    "Metro_Distance_km",
    "Highway_Distance_km",
    "School_Distance_km",
    "Hospital_Distance_km",
    "Road_Development",
    "Metro_Development",
    "Commercial_Development",
    "Infrastructure_Development_Year",
    "Infrastructure_Level",
    "Current_Price"
]

target = "Future_Price_12M"


X = data[features]
y = data[target]


# =========================================================
# PREPROCESSING
# =========================================================

categorical_features = [
    "City"
]

numerical_features = [
    "Year",
    "Month",
    "Area_sqft",
    "Bedrooms",
    "Metro_Distance_km",
    "Highway_Distance_km",
    "School_Distance_km",
    "Hospital_Distance_km",
    "Road_Development",
    "Metro_Development",
    "Commercial_Development",
    "Infrastructure_Development_Year",
    "Infrastructure_Level",
    "Current_Price"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "city",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numerical_features
        )
    ]
)


# =========================================================
# MODEL FACTORY
# =========================================================

def create_model():

    random_forest = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", random_forest)
        ]
    )


# =========================================================
# TIME-BASED EVALUATION
# =========================================================

train_data = data[
    data["Year"] <= 2024
].copy()

test_data = data[
    data["Year"] == 2025
].copy()


if train_data.empty:
    raise ValueError(
        "No training data found before 2025."
    )

if test_data.empty:
    raise ValueError(
        "No 2025 test data found."
    )


X_train = train_data[features]
y_train = train_data[target]

X_test = test_data[features]
y_test = test_data[target]


print("\n========== TIME-BASED EVALUATION ==========")

print(
    "Training period: 2020-2024"
)

print(
    "Testing period : 2025"
)

print(
    "Training records:",
    len(X_train)
)

print(
    "Testing records:",
    len(X_test)
)


# =========================================================
# TRAIN EVALUATION MODEL
# =========================================================

evaluation_model = create_model()

print("\nTraining evaluation model...")

evaluation_model.fit(
    X_train,
    y_train
)


# =========================================================
# EVALUATE
# =========================================================

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


print("\n========== TEST PERFORMANCE ==========")

print(
    f"MAE  : ₹{mae:,.2f}"
)

print(
    f"RMSE : ₹{rmse:,.2f}"
)

print(
    f"R²   : {r2:.4f}"
)


# =========================================================
# SAVE EVALUATION MODEL
# =========================================================

joblib.dump(
    evaluation_model,
    EVALUATION_MODEL_PATH
)

print(
    "\nEvaluation model saved:"
)

print(
    EVALUATION_MODEL_PATH
)


# =========================================================
# TRAIN PRODUCTION MODEL
# =========================================================

print(
    "\n=============================================="
)

print(
    "Training production model on ALL available data..."
)

print(
    "=============================================="
)

production_model = create_model()

production_model.fit(
    X,
    y
)


# =========================================================
# SAVE PRODUCTION MODEL
# =========================================================

joblib.dump(
    production_model,
    MODEL_PATH
)

print(
    "\nProduction model saved:"
)

print(
    MODEL_PATH
)


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

rf_model = production_model.named_steps[
    "model"
]

fitted_preprocessor = (
    production_model
    .named_steps["preprocessor"]
)

feature_names = (
    fitted_preprocessor
    .get_feature_names_out()
)

importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)


# =========================================================
# SAVE METADATA
# =========================================================

metadata = {
    "algorithm": "Random Forest Regression",
    "forecast_horizon": "12 months",
    "target": target,
    "training_period": "All available records",
    "evaluation_period": "2025",
    "mae": float(mae),
    "rmse": float(rmse),
    "r2_score": float(r2),
    "records": int(len(data)),
    "features": features
}

joblib.dump(
    metadata,
    "models/model_metadata.pkl"
)


# =========================================================
# RESULTS
# =========================================================

print(
    "\n========== TOP MODEL FEATURES =========="
)

print(
    importance.head(10).to_string(
        index=False
    )
)


print(
    "\n========== PROJECT STATUS =========="
)

print(
    "Evaluation model : READY"
)

print(
    "Production model : READY"
)

print(
    "Metadata          : READY"
)

print(
    "\n=============================================="
)

print(
    "MODEL BUILD COMPLETED SUCCESSFULLY"
)

print(
    "=============================================="
)