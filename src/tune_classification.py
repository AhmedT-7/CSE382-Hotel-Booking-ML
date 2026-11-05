from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedGroupKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from cleaning import clean_data
from preprocessing import prepare_supervised_data


project_dir = Path(__file__).resolve().parent.parent
df = pd.read_csv(project_dir / "data" / "H2.csv")
df = clean_data(df)

splits, preprocessor = prepare_supervised_data(df, "classification")

models = {
    "Logistic Regression": {
        "classifier": LogisticRegression(max_iter=1000, random_state=42),
        "parameters": {
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__class_weight": [None, "balanced"],
        },
    },
    "Random Forest": {
        "classifier": RandomForestClassifier(random_state=42, n_jobs=-1),
        "parameters": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 20],
            "classifier__min_samples_leaf": [1, 5],
            "classifier__class_weight": [None, "balanced_subsample"],
        },
    },
    "RBF SVM": {
        "classifier": CalibratedClassifierCV(
            estimator=SVC(kernel="rbf", random_state=42),
            cv=3,
            ensemble=False,
        ),
        "parameters": {
            "classifier__estimator__C": [0.1, 1.0, 10.0],
            "classifier__estimator__gamma": ["scale", "auto"],
            "classifier__estimator__class_weight": [None, "balanced"],
        },
    },
}

for name, settings in models.items():
    print(f"\n===== Tuning {name} =====")

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

    groups = pd.util.hash_pandas_object(
        X_train.reset_index(drop=True),
        index=False,
    )

    cv = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=42,
    )

    pipeline = Pipeline(steps=[
        ("preprocessing", clone(preprocessor)),
        ("classifier", settings["classifier"]),
    ])

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=settings["parameters"],
        n_iter=6,
        scoring="f1",
        cv=cv,
        random_state=42,
        n_jobs=1,
        verbose=1,
    )

    search.fit(X_train, y_train, groups=groups)

    best_model = search.best_estimator_
    actual = splits["y_val"]
    predicted = best_model.predict(splits["X_val"])
    scores = best_model.predict_proba(splits["X_val"])[:, 1]

    print("Best cross-validation F1:", search.best_score_)
    print("Best settings:", search.best_params_)
    print("Validation accuracy:", accuracy_score(actual, predicted))
    print("Validation ROC-AUC:", roc_auc_score(actual, scores))
    print("Validation confusion matrix:")
    print(confusion_matrix(actual, predicted))
    print("Validation detailed results:")
    print(classification_report(actual, predicted))