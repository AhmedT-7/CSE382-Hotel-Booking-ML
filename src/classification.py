from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from cleaning import clean_data
from preprocessing import prepare_supervised_data


project_dir = Path(__file__).resolve().parent.parent
df = pd.read_csv(project_dir / "data" / "H2.csv")
df = clean_data(df)

splits, preprocessor = prepare_supervised_data(df, "classification")

model = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000)),
])

model.fit(splits["X_train"], splits["y_train"])

actual = splits["y_val"]
predicted = model.predict(splits["X_val"])
probabilities = model.predict_proba(splits["X_val"])[:, 1]

print("Accuracy:", accuracy_score(actual, predicted))
print("ROC-AUC:", roc_auc_score(actual, probabilities))
print("Confusion matrix:")
print(confusion_matrix(actual, predicted))
print("Detailed results:")
print(classification_report(actual, predicted))