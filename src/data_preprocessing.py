import pandas as pd

# ==========================================
# URBAN PROPERTY PRICE PREDICTION PROJECT
# Data Preprocessing
# ==========================================

# Load the dataset
data = pd.read_csv("data/property_data.csv")

# Display first 5 records
print("\n========== FIRST 5 RECORDS ==========")
print(data.head())

# Display dataset shape
print("\n========== DATASET SIZE ==========")
print("Rows:", data.shape[0])
print("Columns:", data.shape[1])

# Display column names
print("\n========== COLUMN NAMES ==========")
print(data.columns.tolist())

# Display dataset information
print("\n========== DATASET INFORMATION ==========")
data.info()

# Check missing values
print("\n========== MISSING VALUES ==========")
print(data.isnull().sum())

# Check duplicate records
print("\n========== DUPLICATE RECORDS ==========")
print("Number of duplicates:", data.duplicated().sum())

# Statistical summary
print("\n========== STATISTICAL SUMMARY ==========")
print(data.describe())

# Check data types
print("\n========== DATA TYPES ==========")
print(data.dtypes)

print("\n==========================================")
print("DATA PREPROCESSING COMPLETED SUCCESSFULLY")
print("==========================================")