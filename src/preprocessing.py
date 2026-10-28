
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from cleaning import clean_data
from features import make_features


RANDOM_STATE = 42


def split_data(X, y, groups):
    splitter1 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=RANDOM_STATE
    )

    train_idx, remaining_idx = next(
        splitter1.split(X, y, groups=groups)
    )

    splitter2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=RANDOM_STATE
    )

    remaining_groups = groups.iloc[remaining_idx]

    val_local, test_local = next(
        splitter2.split(
            X.iloc[remaining_idx],
            groups=remaining_groups
        )
    )

    val_idx = remaining_idx[val_local]
    test_idx = remaining_idx[test_local]

    return {
        "X_train": X.iloc[train_idx].copy(),
        "X_val": X.iloc[val_idx].copy(),
        "X_test": X.iloc[test_idx].copy(),
        "y_train": y.iloc[train_idx].copy(),
        "y_val": y.iloc[val_idx].copy(),
        "y_test": y.iloc[test_idx].copy()
    }


def build_preprocessor(X):
    numerical_columns = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = [
        col for col in X.columns
        if col not in numerical_columns
    ]

    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="Missing"
            )
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_columns),
            ("cat", categorical_pipeline, categorical_columns)
        ]
    )

    return preprocessor


def prepare_supervised_data(df, task):
    if task not in ["regression", "classification"]:
        raise ValueError("Use regression or classification")

    X, y = make_features(df, task)

    valid = y.notna()
    X = X.loc[valid].copy()
    y = y.loc[valid].copy()

    groups = pd.util.hash_pandas_object(
        df.loc[valid].reset_index(drop=True),
        index=False
    )

    groups.index = X.index

    splits = split_data(X, y, groups)

    preprocessor = build_preprocessor(splits["X_train"])

    return splits, preprocessor


def main():
    project_dir = Path(__file__).resolve().parent.parent
    dataset_path = project_dir / "data" / "H2.csv"

    print("Loading hotel booking dataset...")

    df = pd.read_csv(dataset_path)
    df = clean_data(df)

    print("Dataset shape:", df.shape)

    for task in ["regression", "classification"]:
        print("\n-----------------------------")
        print("Task:", task)
        print("-----------------------------")

        splits, preprocessor = prepare_supervised_data(
            df, task
        )

        X_train = preprocessor.fit_transform(
            splits["X_train"]
        )

        X_val = preprocessor.transform(
            splits["X_val"]
        )

        X_test = preprocessor.transform(
            splits["X_test"]
        )

        print("Training shape:", X_train.shape)
        print("Validation shape:", X_val.shape)
        print("Testing shape:", X_test.shape)

        if task == "classification":
            print("\nCancellation rates:")

            for name in ["train", "val", "test"]:
                rate = splits["y_" + name].mean() * 100
                print(name, ":", round(rate, 2), "%")

        print("Preprocessing completed successfully!")

    print("\nAll preprocessing tasks completed.")


if __name__ == "__main__":
    main()
