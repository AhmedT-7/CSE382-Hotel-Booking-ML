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


   ## RBF SVM baseline

Validation results; untuned; test set not used. Trained on a stratified subset of 5,000 training rows.

- Accuracy: 0.7748
- ROC-AUC: 0.8146
- Cancellation precision: 0.98
- Cancellation recall: 0.47
- Cancellation F1-score: 0.64
- Confusion matrix:
[[6937, 52],
   [2662, 2403]]