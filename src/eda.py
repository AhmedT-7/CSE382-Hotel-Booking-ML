
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

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

# Display all column names
print("\n8. Column Names:")
print(df.columns.tolist())

# Summary statistics for categorical columns
print("\n9. Categorical Summary:")
print(df.describe(include=["object", "str"]))

# Missing values analysis
missing = df.isnull().sum()

missing_percentage = (missing / len(df)) * 100

missing_table = pd.DataFrame({
    "Missing Count": missing,
    "Missing Percentage": missing_percentage
})

print("\n10. Missing Value Analysis:")
print(missing_table[missing_table["Missing Count"] > 0])

# Duplicate analysis
duplicate_count = df.duplicated().sum()
duplicate_percentage = duplicate_count / len(df) * 100

print("\n11. Duplicate Analysis:")
print("Duplicate rows:", duplicate_count)
print(f"Duplicate percentage: {duplicate_percentage:.2f}%")

print("\n12. Invalid Value Analysis:")

columns_to_check = [
    "ADR",
    "LeadTime",
    "Adults",
    "Children",
    "Babies",
    "StaysInWeekendNights",
    "StaysInWeekNights"
]

for column in columns_to_check:
    negative_count = (df[column] < 0).sum()

    print(f"{column}: {negative_count} negative values")

# Check bookings with no guests
zero_guests = (
    (df["Adults"] == 0) &
    (df["Children"].fillna(0) == 0) &
    (df["Babies"] == 0)
)

print("Bookings with zero guests:", zero_guests.sum())

print("\n13. Outlier Analysis:")

columns = [
    "ADR",
    "LeadTime",
    "StaysInWeekendNights",
    "StaysInWeekNights"
]

for column in columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    mean = df[column].mean()
    median = df[column].median()

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ]

    print(f"\n{column}:")
    print(f"Minimum: {df[column].min():.2f}")
    print(f"Maximum: {df[column].max():.2f}")
    print(f"Mean: {mean:.2f}")
    print(f"Median: {median:.2f}")
    print(f"Q1: {Q1:.2f}")
    print(f"Q3: {Q3:.2f}")
    print(f"IQR: {IQR:.2f}")
    print(f"Lower Bound: {lower_bound:.2f}")
    print(f"Upper Bound: {upper_bound:.2f}")
    print(f"Outliers Detected: {len(outliers)}")

plt.figure(figsize=(10, 5))

plt.hist(df.loc[df["ADR"].between(0, 300), "ADR"], bins=50)

plt.title("Distribution of Average Daily Rate")
plt.xlabel("ADR")
plt.ylabel("Number of Bookings")

plt.tight_layout()
plt.show()

# Booking Cancellation Distribution
cancellation_counts = df["IsCanceled"].value_counts().sort_index()

plt.figure(figsize=(8, 5))

plt.bar(
    ["Not Canceled", "Canceled"],
    cancellation_counts.values
)

plt.title("Hotel Booking Cancellation Distribution")
plt.xlabel("Booking Status")
plt.ylabel("Number of Bookings")

plt.tight_layout()
plt.show()

print("\nCancellation Percentages:")
print(df["IsCanceled"].value_counts(normalize=True) * 100)

# Lead Time Distribution
plt.figure(figsize=(10, 5))

plt.hist(df["LeadTime"], bins=50)

plt.title("Hotel Booking Lead Time Distribution")
plt.xlabel("Lead Time (Days)")
plt.ylabel("Number of Bookings")

plt.tight_layout()
plt.show()

# Lead Time vs Cancellation

plt.figure(figsize=(9, 5))

plt.boxplot([
    df[df["IsCanceled"] == 0]["LeadTime"],
    df[df["IsCanceled"] == 1]["LeadTime"]
], tick_labels=["Not Canceled", "Canceled"])

plt.title("Lead Time vs Booking Cancellation")
plt.xlabel("Booking Status")
plt.ylabel("Lead Time (Days)")

plt.tight_layout()
plt.show()
print("\nLead Time by Cancellation Status:")

for status in [0, 1]:
    lead_times = df[df["IsCanceled"] == status]["LeadTime"]

    print(f"\nStatus {status}:")
    print(f"Mean: {lead_times.mean():.2f}")
    print(f"Median: {lead_times.median():.2f}")
    print(f"Q1: {lead_times.quantile(0.25):.2f}")
    print(f"Q3: {lead_times.quantile(0.75):.2f}")

# Unique Values in Each Column
print("\nUnique Values in Each Column:")

categorical_columns = df.select_dtypes(
    include=["object", "str"]
).columns

for column in categorical_columns:
    unique_count = df[column].nunique(dropna=False)
    print(f"{column}: {unique_count} unique values")
