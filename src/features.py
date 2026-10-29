
from pathlib import Path
import pandas as pd

from cleaning import clean_data


COMMON_FEATURES = [
    "LeadTime",
    "ArrivalDateYear",
    "ArrivalDateMonth",
    "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "Meal",
    "MarketSegment",
    "DistributionChannel",
    "IsRepeatedGuest",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
    "ReservedRoomType",
    "DepositType",
    "CustomerType"
]

REGRESSION_FEATURES = COMMON_FEATURES + [
    "Country"
]

CLASSIFICATION_FEATURES = COMMON_FEATURES


def make_features(df, task):
    if task not in ["regression", "classification"]:
        raise ValueError("Invalid task")

    data = df.copy()
    data.columns = data.columns.str.strip()

    if task == "regression":
        columns = REGRESSION_FEATURES
        target = "ADR"
    else:
        columns = CLASSIFICATION_FEATURES
        target = "IsCanceled"

    missing = [
        col for col in columns + [target]
        if col not in data.columns
    ]

    if missing:
        raise ValueError("Missing columns: " + str(missing))

    X = data[columns].copy()
    X = X.replace("NULL", pd.NA)

    X["TotalNights"] = (
        X["StaysInWeekendNights"] +
        X["StaysInWeekNights"]
    )

    X["TotalGuests"] = (
        X["Adults"] +
        X["Children"] +
        X["Babies"]
    )

    y = data[target].copy()

    return X, y


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent.parent

    df = pd.read_csv(project_dir / "data" / "H2.csv")
    df = clean_data(df)

    for task in ["regression", "classification"]:
        X, y = make_features(df, task)

        print("\nTask:", task)
        print("Feature shape:", X.shape)
        print("Target:", y.name)
        print("Features:", X.columns.tolist())
