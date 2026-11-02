# Classification Baseline

- Model: Logistic Regression
- Run command: `.\.venv\Scripts\python.exe src\classification.py`
- Results are from the validation set.
- The test set was not used.

- Accuracy: 0.7724
- ROC-AUC: 0.8199
- Confusion matrix:
[[6771, 218],
   [2526, 2539]]

## Random Forest baseline

Validation results; untuned; test set not used.

- Accuracy: 0.7679
- ROC-AUC: 0.8311
- Cancellation precision: 0.76
- Cancellation recall: 0.65
- Cancellation F1-score: 0.70
- Confusion matrix:
[[5943, 1046],
   [1752, 3313]]