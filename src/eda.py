
from pathlib import Path
import pandas as pd

# Locate the dataset
project_dir = Path(__file__).resolve().parent.parent
data_path = project_dir / "data" / "H2.csv"

# Load dataset
df = pd.read_csv(data_path)

print("===== HOTEL BOOKING DATASET =====")

# Dataset dimensions
print("\n1. Dataset Shape:")
print(df.shape)

# First five bookings
print("\n2. First 5 Rows:")
print(df.head())

# Data types
print("\n3. Column Data Types:")
print(df.dtypes)

# Missing values
print("\n4. Missing Values:")
print(df.isnull().sum())

# Duplicate rows
print("\n5. Duplicate Rows:")
print(df.duplicated().sum())

# Numerical statistics
print("\n6. Summary Statistics:")
print(df.describe())

# Cancellation distribution
print("\n7. Cancellation Distribution:")
print(df["IsCanceled"].value_counts())

print("\nEDA COMPLETED SUCCESSFULLY!")
