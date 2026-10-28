
from pathlib import Path
import numpy as np
import pandas as pd

from cleaning import clean_data


BOOKING_FEATURES = [
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
    "Country",
    "MarketSegment",
    "DistributionChannel",
    "IsRepeatedGuest",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
    "ReservedRoomType",
    "DepositType",
    "CustomerType",
]


def make_features(df, task):
    if task not in ["regression", "classification", "clustering"]:
        raise ValueError("Invalid task")

    data = df.copy()
    data.columns = data.columns.str.strip()

    missing = [
        col for col in BOOKING_FEATURES
        if col not in data.columns
    ]

    if missing:
        raise ValueError("Missing columns: " + str(missing))

    X = data[BOOKING_FEATURES].copy()
    X = X.replace("NULL", np.nan)

    X["TotalNights"] = (
        X["StaysInWeekendNights"] +
        X["StaysInWeekNights"]
    )

    X["TotalGuests"] = (
        X["Adults"] +
        X["Children"] +
        X["Babies"]
    )

    X["LogLeadTime"] = np.log1p(X["LeadTime"])

    if task == "regression":
        if "ADR" not in data.columns:
            raise ValueError("ADR target missing")

        y = data["ADR"].copy()

    elif task == "classification":
        if "IsCanceled" not in data.columns:
            raise ValueError("IsCanceled target missing")

        y = data["IsCanceled"].copy()

        if "ADR" in data.columns:
            X["ADR"] = data["ADR"]

    else:
        y = None

        if "ADR" in data.columns:
            X["ADR"] = data["ADR"]

    return X, y


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent.parent

    df = pd.read_csv(project_dir / "data" / "H2.csv")
    df = clean_data(df)

    for task in ["regression", "classification", "clustering"]:
        X, y = make_features(df, task)

        print("\nTask:", task)
        print("Feature shape:", X.shape)
        print("Number of features:", len(X.columns))

        if y is not None:
            print("Target:", y.name)

        print("Features:", X.columns.tolist())
