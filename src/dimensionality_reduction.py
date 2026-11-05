"""Task 6: leakage-safe PCA and LDA experiments on H2 bookings.

Run: python src/dimensionality_reduction.py
Requires: pandas, numpy, scikit-learn, matplotlib.
Results are generated under outputs/dimensionality_reduction/.
"""

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "H2.csv"
OUT = ROOT / "outputs" / "dimensionality_reduction"
RANDOM_STATE = 42

# Match the booking-time feature policy in the team's src/features.py.
COMMON = [
    "LeadTime", "ArrivalDateYear", "ArrivalDateMonth", "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth", "StaysInWeekendNights", "StaysInWeekNights",
    "Adults", "Children", "Babies", "Meal", "MarketSegment",
    "DistributionChannel", "IsRepeatedGuest", "PreviousCancellations",
    "PreviousBookingsNotCanceled", "ReservedRoomType", "DepositType",
    "CustomerType",
]


def load_features(task):
    """Only use information available when a booking is made."""
    df = pd.read_csv(DATA)
    df.columns = df.columns.str.strip()
    df["Children"] = pd.to_numeric(df["Children"], errors="coerce").fillna(0)
    columns = COMMON + (["Country"] if task == "regression" else [])
    target = "ADR" if task == "regression" else "IsCanceled"
    data = df[columns + [target]].copy().replace("NULL", np.nan)
    for column in data.select_dtypes(include=["object"]).columns:
        data[column] = data[column].str.strip()
    data = data.dropna(subset=[target]).reset_index(drop=True)
    X = data[columns].copy()
    X["TotalNights"] = X["StaysInWeekendNights"] + X["StaysInWeekNights"]
    X["TotalGuests"] = X["Adults"] + X["Children"] + X["Babies"]
    y = data[target]
    # Group identical bookings together to minimize duplicate-row leakage.
    groups = pd.util.hash_pandas_object(data, index=False)
    return X, y, groups


def split(X, y, groups):
    first = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_STATE)
    tr, rest = next(first.split(X, y, groups))
    second = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_STATE)
    va_local, te_local = next(second.split(X.iloc[rest], y.iloc[rest], groups.iloc[rest]))
    return tr, rest[va_local], rest[te_local]


def preprocessor(X):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                           ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
                           ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ], sparse_threshold=0.0)


def make_pipeline(task, X_train, reducer=None):
    estimator = (LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
                 if task == "classification" else Ridge(alpha=1.0))
    steps = [("preprocessing", preprocessor(X_train))]
    if reducer is not None:
        steps.append(("reduction", reducer))
    steps.append(("model", estimator))
    return Pipeline(steps)


def evaluate(pipe, X_train, y_train, X_val, y_val, task, label):
    start = time.perf_counter()
    pipe.fit(X_train, y_train)
    seconds = time.perf_counter() - start
    predicted = pipe.predict(X_val)
    result = {"task": task, "method": label, "train_seconds": round(seconds, 3),
              "original_dimensions": int(pipe.named_steps["preprocessing"].transform(X_train.iloc[:1]).shape[1]),
              "retained_dimensions": int(pipe.named_steps["reduction"].n_components_)
              if "reduction" in pipe.named_steps and label == "PCA" else
              (1 if label == "LDA" else int(pipe.named_steps["preprocessing"].transform(X_train.iloc[:1]).shape[1]))}
    if task == "classification":
        probabilities = pipe.predict_proba(X_val)[:, 1]
        result.update(accuracy=round(float(accuracy_score(y_val, predicted)), 5),
                      f1=round(float(f1_score(y_val, predicted)), 5),
                      roc_auc=round(float(roc_auc_score(y_val, probabilities)), 5))
    else:
        result.update(mae=round(float(mean_absolute_error(y_val, predicted)), 5),
                      rmse=round(float(np.sqrt(mean_squared_error(y_val, predicted))), 5),
                      r2=round(float(r2_score(y_val, predicted)), 5))
    return result, pipe


def plot_variance(model, path):
    ratios = model.named_steps["reduction"].explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.arange(1, len(ratios) + 1), np.cumsum(ratios), linewidth=2)
    ax.axhline(0.95, linestyle="--", color="tomato", label="95% variance")
    ax.set(xlabel="Number of principal components", ylabel="Cumulative explained variance",
           title="PCA explained variance")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    metadata = {"random_state": RANDOM_STATE, "evaluation_split": "validation",
                "test_set_used": False, "pca_variance_threshold": 0.95,
                "prediction_time": "booking creation"}
    for task in ("classification", "regression"):
        print(f"\n=== {task.upper()} ===", flush=True)
        X, y, groups = load_features(task)
        train, val, test = split(X, y, groups)
        X_train, X_val = X.iloc[train], X.iloc[val]
        y_train, y_val = y.iloc[train], y.iloc[val]
        metadata[task] = {"train_rows": len(train), "validation_rows": len(val),
                          "held_out_test_rows": len(test)}
        variants = [("Original", None),
                    ("PCA", PCA(n_components=0.95, svd_solver="full"))]
        if task == "classification":
            # Binary LDA yields at most one discriminant component.
            variants.append(("LDA", LinearDiscriminantAnalysis(n_components=1,
                                                                solver="svd")))
        for label, reducer in variants:
            pipeline = make_pipeline(task, X_train, clone(reducer) if reducer is not None else None)
            result, fitted = evaluate(pipeline, X_train, y_train, X_val, y_val, task, label)
            results.append(result)
            print(result, flush=True)
            if label == "PCA":
                plot_variance(fitted, OUT / f"{task}_pca_explained_variance.png")
    (OUT / "experiment_metrics.json").write_text(
        json.dumps({"metadata": metadata, "results": results}, indent=2), encoding="utf-8")
    pd.DataFrame(results).to_csv(OUT / "experiment_comparison.csv", index=False)
    print(f"\nSaved results to {OUT}")


if __name__ == "__main__":
    main()
