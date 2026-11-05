
from pathlib import Path
import pandas as pd


def clean_data(df, remove_zero_guests=False):
    # Create a copy to protect original data
    cleaned_df = df.copy()

    # 1. Remove spaces from column names
    cleaned_df.columns = cleaned_df.columns.str.strip()

    # 2. Remove spaces from text values
    text_columns = cleaned_df.select_dtypes(
        include=["object", "str"]
    ).columns

    for column in text_columns:
        cleaned_df[column] = cleaned_df[column].str.strip()

    # 3. Fill the 4 missing Children values with 0
    cleaned_df["Children"] = cleaned_df["Children"].fillna(0)

    # 4. Convert "NULL" text into actual missing values
    for column in ["Agent", "Company"]:
        cleaned_df[column] = cleaned_df[column].replace(
            "NULL", pd.NA
        )

    # 5. Identify bookings with zero guests
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

    # Check missing values after cleaning
    print("\nMissing Values After Cleaning:")
    print(cleaned_df[["Children", "Agent", "Company"]].isna().sum())

    # Check if spaces were removed
    print("\nMeal Categories After Cleaning:")
    print(cleaned_df["Meal"].unique())

    # Test optional removal
    filtered_df = clean_data(df, remove_zero_guests=True)

    print("\nShape after optional filtering:", filtered_df.shape)
    print("Rows excluded:", len(df) - len(filtered_df))
