
# import pandas as pd

# def clean_data(df):
#     # Create a copy to protect the original data
#     cleaned_df = df.copy()

#     # Remove accidental spaces from column names
#     cleaned_df.columns = cleaned_df.columns.str.strip()

#     return cleaned_df


# if __name__ == "__main__":
#     from pathlib import Path

#     project_dir = Path(__file__).resolve().parent.parent

#     df = pd.read_csv(project_dir / "data" / "H2.csv")
#     cleaned_df = clean_data(df)

#     print("Original shape:", df.shape)
#     print("Cleaned shape:", cleaned_df.shape)
#     print("Column names:", cleaned_df.columns.tolist())

#     # Investigate bookings with zero guests
#     zero_guests = cleaned_df[
#         (cleaned_df["Adults"] == 0) &
#         (cleaned_df["Children"].fillna(0) == 0) &
#         (cleaned_df["Babies"] == 0)
#     ]

#     print("\nZero Guest Bookings:", len(zero_guests))

#     print("\nCancellation Status:")
#     print(zero_guests["IsCanceled"].value_counts())

#     print("\nReservation Status:")
#     print(zero_guests["ReservationStatus"].value_counts())

#     print("\nAll Zero Guest Bookings:")
#     print(zero_guests[[
#         "Adults",
#         "Children",
#         "Babies",
#         "IsCanceled",
#         "ReservationStatus",
#         "ADR"
#     ]].to_string())

# def clean_data(df, remove_zero_guests=False):
#     cleaned_df = df.copy()

#     # Remove accidental spaces from column names
#     cleaned_df.columns = cleaned_df.columns.str.strip()

#     # Identify bookings with zero guests
#     zero_guests = (
#         (cleaned_df["Adults"] == 0) &
#         (cleaned_df["Children"] == 0) &
#         (cleaned_df["Babies"] == 0)
#     )

#     # Remove only if explicitly requested
#     if remove_zero_guests:
#         cleaned_df = cleaned_df.loc[~zero_guests].copy()

#     return cleaned_df
# filtered_df = clean_data(df, remove_zero_guests=True)

# print("Original shape:", df.shape)
# print("Filtered shape:", filtered_df.shape)
# print("Rows excluded:", len(df) - len(filtered_df))

# # Duplicate investigation
# print("\nDuplicate Investigation:")

# duplicate_count = cleaned_df.duplicated().sum()

# print("Total rows:", len(cleaned_df))
# print("Duplicate rows:", duplicate_count)
# print("Unique rows:", len(cleaned_df) - duplicate_count)

# # Compare cancellations among duplicated and other rows
# duplicate_mask = cleaned_df.duplicated(keep=False)

# print("\nCancellation status of repeated records:")
# print(
#     cleaned_df.loc[duplicate_mask, "IsCanceled"].value_counts()
# )

# # Compare with records appearing only once
# print("\nCancellation status of non-repeated records:")
# print(
#     cleaned_df.loc[~duplicate_mask, "IsCanceled"].value_counts()
# )

# # Missing Values Investigation
# print("\nMissing Values Investigation:")

# columns = ["Children", "Country", "Agent", "Company"]

# for column in columns:
#     print(f"\n{column}:")

#     print("Missing values:", cleaned_df[column].isna().sum())

#     print("NULL text values:",
#           cleaned_df[column].astype("string")
#           .str.strip().str.upper().eq("NULL").sum())

#     print("Most common values:")
#     print(cleaned_df[column].value_counts(dropna=False).head(10))

from pathlib import Path
import pandas as pd


def clean_data(df, remove_zero_guests=False):
    # Create a copy to protect original data
    cleaned_df = df.copy()

    # Remove spaces from column names
    cleaned_df.columns = cleaned_df.columns.str.strip()

    # Identify bookings with zero guests
    zero_guests = (
        (cleaned_df["Adults"] == 0) &
        (cleaned_df["Children"] == 0) &
        (cleaned_df["Babies"] == 0)
    )

    # Optionally remove zero-guest bookings
    if remove_zero_guests:
        cleaned_df = cleaned_df.loc[~zero_guests].copy()

    return cleaned_df


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent.parent

    # Load dataset
    df = pd.read_csv(project_dir / "data" / "H2.csv")

    # Test cleaning without removing bookings
    cleaned_df = clean_data(df)

    print("Original shape:", df.shape)
    print("Cleaned shape:", cleaned_df.shape)

    # Test optional removal
    filtered_df = clean_data(df, remove_zero_guests=True)

    print("Shape after optional filtering:", filtered_df.shape)
    print("Rows excluded:", len(df) - len(filtered_df))
