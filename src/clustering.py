import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from cleaning import clean_data

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "H2.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "clustering"
ARTIFACT_DIR = PROJECT_DIR / "artifacts" / "clustering"

RANDOM_STATE = 42
SILHOUETTE_SAMPLE = 10000

SKEWED_FEATURES = ["LeadTime", "PreviousCancellations", "PreviousBookingsNotCanceled"]
COUNT_FEATURES = ["StaysInWeekendNights", "StaysInWeekNights", "Adults", "Kids", "IsRepeatedGuest"]
CATEGORICAL_FEATURES = ["MarketSegment", "DistributionChannel", "CustomerType", "DepositType", "Meal"]
CLUSTER_FEATURES = SKEWED_FEATURES + COUNT_FEATURES + CATEGORICAL_FEATURES

EXCLUDED_FEATURES = {
    "IsCanceled": "outcome of the booking",
    "ReservationStatus": "outcome of the booking",
    "ReservationStatusDate": "set after the outcome",
    "ADR": "regression target, used only to describe clusters",
    "AssignedRoomType": "known only at check-in",
    "BookingChanges": "changes after booking",
    "DaysInWaitingList": "changes after booking",
    "TotalOfSpecialRequests": "can change after booking",
    "RequiredCarParkingSpaces": "can change after booking",
    "Country": "often filled at check-in, high cardinality",
    "Agent": "ID with many levels",
    "Company": "ID, about 95% missing",
}


def load_data(path=DATA_PATH):
    df = clean_data(pd.read_csv(path), remove_zero_guests=True).reset_index(drop=True)
    df["Kids"] = df["Children"] + df["Babies"]
    return df


def build_preprocessor():
    skewed = Pipeline([
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([
        ("skewed", skewed, SKEWED_FEATURES),
        ("counts", StandardScaler(), COUNT_FEATURES),
        ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])


def build_pipeline(n_clusters=4):
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("kmeans", KMeans(n_clusters=n_clusters, n_init=10, random_state=RANDOM_STATE)),
    ])


def evaluate(X, labels):
    return {
        "silhouette": float(silhouette_score(X, labels, sample_size=SILHOUETTE_SAMPLE, random_state=RANDOM_STATE)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
    }


def profile_clusters(df, labels):
    profile = df.assign(Cluster=labels).groupby("Cluster").agg(
        Size=("LeadTime", "size"),
        LeadTime=("LeadTime", "median"),
        WeekendNights=("StaysInWeekendNights", "mean"),
        WeekNights=("StaysInWeekNights", "mean"),
        Adults=("Adults", "mean"),
        Kids=("Kids", "mean"),
        RepeatedGuest=("IsRepeatedGuest", "mean"),
        TopSegment=("MarketSegment", lambda s: s.mode()[0]),
        TopCustomerType=("CustomerType", lambda s: s.mode()[0]),
        NonRefundShare=("DepositType", lambda s: (s == "Non Refund").mean()),
        MedianADR=("ADR", "median"),
        CancelRate=("IsCanceled", "mean"),
    )
    profile["Share"] = profile["Size"] / profile["Size"].sum()
    return profile.round(3)


def plot_pca(X, labels, path):
    points = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(points), size=min(15000, len(points)), replace=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(points[idx, 0], points[idx, 1], c=labels[idx], cmap="tab10", s=4, alpha=0.6)
    ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("K-Means baseline clusters (PCA 2D)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_baseline(n_clusters=4):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    pipeline = build_pipeline(n_clusters)

    start = time.perf_counter()
    labels = pipeline.fit_predict(df[CLUSTER_FEATURES])
    train_time = time.perf_counter() - start

    X = pipeline.named_steps["preprocess"].transform(df[CLUSTER_FEATURES])
    metrics = evaluate(X, labels)
    results = {
        "algorithm": "KMeans",
        "config": "baseline",
        "n_clusters": n_clusters,
        "n_rows": len(df),
        "n_features_after_encoding": X.shape[1],
        "train_time_sec": round(train_time, 2),
        **metrics,
        "features": CLUSTER_FEATURES,
        "excluded": EXCLUDED_FEATURES,
    }

    profile = profile_clusters(df, labels)
    profile.to_csv(OUTPUT_DIR / "kmeans_baseline_profile.csv")
    plot_pca(X, labels, OUTPUT_DIR / "kmeans_baseline_pca.png")
    (OUTPUT_DIR / "kmeans_baseline_metrics.json").write_text(json.dumps(results, indent=2))
    joblib.dump(pipeline, ARTIFACT_DIR / "kmeans_baseline.joblib")

    print(f"Rows: {len(df)}  Encoded features: {X.shape[1]}  k={n_clusters}")
    print(f"Silhouette: {metrics['silhouette']:.3f}  Davies-Bouldin: {metrics['davies_bouldin']:.3f}  Time: {train_time:.1f}s")
    print(profile.to_string())
    return results


if __name__ == "__main__":
    run_baseline()
