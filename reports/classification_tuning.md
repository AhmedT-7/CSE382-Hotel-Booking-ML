# Cancellation Classifier Tuning

All results below are from the validation set. The test set was not used.
Tuning used 3-fold cross-validation, F1-score, and 6 sampled settings per model.
The RBF SVM used a stratified subset of 5,000 training rows.

| Model | Best CV F1 | Validation accuracy | Validation ROC-AUC | Cancellation precision | Cancellation recall | Cancellation F1 |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6832 | 0.7684 | 0.8202 | 0.80 | 0.59 | 0.68 |
| Random Forest | 0.7117 | 0.7759 | 0.8529 | 0.77 | 0.67 | 0.72 |
| RBF SVM | 0.6775 | 0.7591 | 0.8200 | 0.76 | 0.63 | 0.69 |

## Best settings found

- Logistic Regression: class_weight=balanced, C=10.0
- Random Forest: n_estimators=100, min_samples_leaf=5, max_depth=20, class_weight=balanced_subsample
- RBF SVM: gamma=scale, class_weight=balanced, C=10.0

## Confusion matrices

Rows are actual classes; columns are predicted classes.

- Logistic Regression: [[6259, 730], [2062, 3003]]
- Random Forest: [[5953, 1036], [1665, 3400]]
- RBF SVM: [[5969, 1020], [1884, 3181]]

## Current comparison

Random Forest has the highest validation F1-score and ROC-AUC among these tuned models.
This is a validation-based result; final selection and test evaluation are still to come.