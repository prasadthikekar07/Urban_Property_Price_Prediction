import os
import sys
import pandas as pd
import joblib


# =========================================================
# PROJECT VALIDATION
# =========================================================

print("\n==============================================")
print(" URBAN PROPERTY PRICE FORECASTING")
print(" PROJECT VALIDATION")
print("==============================================\n")


errors = []
passed = 0


# =========================================================
# REQUIRED FILES
# =========================================================

required_files = [
    "app.py",
    "README.md",
    "requirements.txt",
    "run_project.bat",

    "data/property_data.csv",

    "models/property_price_model.pkl",
    "models/evaluation_model.pkl",
    "models/model_metadata.pkl",

    "src/data_preprocessing.py",
    "src/generate_dataset.py",
    "src/train_model.py",
    "src/predict.py",
    "src/visualize.py"
]


print("Checking project files...\n")

for file_path in required_files:

    if os.path.exists(file_path):

        print(f"[PASS] {file_path}")
        passed += 1

    else:

        print(f"[FAIL] {file_path}")
        errors.append(
            f"Missing file: {file_path}"
        )


# =========================================================
# DATASET CHECK
# =========================================================

print("\nChecking dataset...\n")

try:

    data = pd.read_csv(
        "data/property_data.csv"
    )

    required_columns = [
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
        "Current_Price",
        "Future_Price_12M",
        "Price_Growth_Percent",
        "Price_Trend"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if not missing_columns:

        print(
            "[PASS] Dataset columns are correct"
        )

        print(
            f"[PASS] Dataset records: {len(data)}"
        )

        print(
            f"[PASS] Cities: {data['City'].nunique()}"
        )

        passed += 3

    else:

        print(
            "[FAIL] Missing columns:"
        )

        for column in missing_columns:
            print(
                "      -",
                column
            )

        errors.append(
            "Dataset has missing columns"
        )

except Exception as error:

    print(
        "[FAIL] Dataset could not be loaded"
    )

    errors.append(
        str(error)
    )


# =========================================================
# MODEL CHECK
# =========================================================

print("\nChecking trained models...\n")

try:

    production_model = joblib.load(
        "models/property_price_model.pkl"
    )

    print(
        "[PASS] Production model loaded"
    )

    passed += 1

except Exception as error:

    print(
        "[FAIL] Production model could not be loaded"
    )

    errors.append(
        str(error)
    )


try:

    evaluation_model = joblib.load(
        "models/evaluation_model.pkl"
    )

    print(
        "[PASS] Evaluation model loaded"
    )

    passed += 1

except Exception as error:

    print(
        "[FAIL] Evaluation model could not be loaded"
    )

    errors.append(
        str(error)
    )


# =========================================================
# METADATA CHECK
# =========================================================

print("\nChecking model metadata...\n")

try:

    metadata = joblib.load(
        "models/model_metadata.pkl"
    )

    required_metadata = [
        "algorithm",
        "forecast_horizon",
        "target",
        "evaluation_period",
        "mae",
        "rmse",
        "r2_score",
        "records",
        "features"
    ]

    missing_metadata = [
        item
        for item in required_metadata
        if item not in metadata
    ]

    if not missing_metadata:

        print(
            "[PASS] Model metadata is valid"
        )

        print(
            "[INFO] Algorithm:",
            metadata["algorithm"]
        )

        print(
            "[INFO] Forecast:",
            metadata["forecast_horizon"]
        )

        print(
            "[INFO] Evaluation:",
            metadata["evaluation_period"]
        )

        passed += 1

    else:

        print(
            "[FAIL] Metadata is incomplete"
        )

        errors.append(
            "Missing metadata fields"
        )

except Exception as error:

    print(
        "[FAIL] Metadata could not be loaded"
    )

    errors.append(
        str(error)
    )


# =========================================================
# SAMPLE PREDICTION CHECK
# =========================================================

print("\nChecking sample prediction...\n")

try:

    sample = data.iloc[[0]].copy()

    features = metadata["features"]

    sample_input = sample[features]

    prediction = production_model.predict(
        sample_input
    )

    print(
        "[PASS] Sample prediction executed"
    )

    print(
        f"[INFO] Sample forecast: "
        f"₹{prediction[0]:,.2f}"
    )

    passed += 1

except Exception as error:

    print(
        "[FAIL] Sample prediction failed"
    )

    errors.append(
        str(error)
    )


# =========================================================
# FINAL RESULT
# =========================================================

print(
    "\n=============================================="
)

print(
    " VALIDATION SUMMARY"
)

print(
    "=============================================="
)

print(
    f"Passed checks: {passed}"
)

print(
    f"Failed checks: {len(errors)}"
)


if errors:

    print(
        "\nProblems found:"
    )

    for error in errors:
        print(
            "-",
            error
        )

    print(
        "\nProject validation FAILED."
    )

    sys.exit(1)

else:

    print(
        "\nALL VALIDATION CHECKS PASSED."
    )

    print(
        "Project is ready for demonstration."
    )

    print(
        "=============================================="
    )

    sys.exit(0)