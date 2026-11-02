from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from cleaning import clean_data
from preprocessing import prepare_supervised_data


project_dir = Path(__file__).resolve().parent.parent
df = pd.read_csv(project_dir / "data" / "H2.csv")
df = clean_data(df)

splits, preprocessor = prepare_supervised_data(df, "classification")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
    ),
    "RBF SVM": SVC(
        kernel="rbf",
        random_state=42,
    ),
}

for name, classifier in models.items():
    print(f"\n===== {name} =====")

    X_train = splits["X_train"]
    y_train = splits["y_train"]

    if name == "RBF SVM":
        X_train, _, y_train, _ = train_test_split(
            X_train,
            y_train,
            train_size=5000,
            stratify=y_train,
            random_state=42,
        )
        print("Training rows used for SVM:", len(y_train))

    model = Pipeline(steps=[
        ("preprocessing", clone(preprocessor)),
        ("classifier", classifier),
    ])

    model.fit(X_train, y_train)

    actual = splits["y_val"]
    predicted = model.predict(splits["X_val"])
    if hasattr(model, "predict_proba"):
     scores = model.predict_proba(splits["X_val"])[:, 1]
    else:
     scores = model.decision_function(splits["X_val"])

    print("Accuracy:", accuracy_score(actual, predicted))
    print("ROC-AUC:", roc_auc_score(actual, scores))
    print("Confusion matrix:")
    print(confusion_matrix(actual, predicted))
    print("Detailed results:")
    print(classification_report(actual, predicted))