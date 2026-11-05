# Task 6 - Dimensionality Reduction and ML Experiments

## Objective
Compare supervised prediction with the original booking-time predictors against reduced representations. PCA is unsupervised and preserves variance; LDA is supervised and maximizes class separation. Binary cancellation classification permits only one LDA component.

## Reproducible method
Run `python src/dimensionality_reduction.py` from the project root. The script reads `data/H2.csv`, applies the same booking-time feature policy as Task 2, adds TotalNights and TotalGuests, uses grouped 70/15/15 train/validation/test splits (random seed 42), imputes missing values, standardizes numeric features, one-hot encodes categories, and fits dimensionality reduction **inside** the training pipeline. No reducer or scaler sees validation or test data while fitting. The test set was not evaluated. PCA retains at least 95% training variance. Logistic Regression is the classification estimator and Ridge is the regression estimator; the comparison holds each task's estimator constant.

## Classification: predict IsCanceled (validation)

| Representation | Dimensions | Accuracy | F1 | ROC-AUC | Fit time (s) |
|---|---:|---:|---:|---:|---:|
| Original | 58 | 0.76357 | 0.62497 | 0.79891 | 1.437 |
| PCA (95%) | 21 | 0.75719 | 0.61841 | 0.78855 | 1.027 |
| LDA | 1 | 0.75114 | 0.62488 | 0.78885 | 1.021 |

PCA reduced the encoded feature count by 63.8%; LDA reduced it by 98.3%. Original features yielded the highest validation accuracy and ROC-AUC. LDA maintained a similar F1 despite compressing to one axis.

## Regression: predict ADR (validation)

| Representation | Dimensions | MAE | RMSE | R2 | Fit time (s) |
|---|---:|---:|---:|---:|---:|
| Original | 213 | 16.96986 | 23.19222 | 0.65124 | 1.425 |
| PCA (95%) | 28 | 20.50503 | 28.19475 | 0.48456 | 3.844 |

PCA reduced the encoded feature count by 86.9%, but raised validation RMSE by about 21.6% and lowered R2. LDA is not applied to a continuous ADR target because conventional LDA requires class labels.

## Interpretation and limitations
- Explained variance is not the same as predictive information. Rare but predictive directions may be discarded by PCA.
- This is a controlled baseline comparison, not an exhaustive hyperparameter search; fitting time is machine-dependent.
- The 70/15/15 grouped split is random, not chronological. A future deployment study should consider temporal drift and confirm feature availability at booking creation.
- The held-out test set remains untouched. Select methods and hyperparameters using validation data, then evaluate the chosen final pipeline once on test.
- These measurements are from Task 6's logistic/Ridge baselines and **must not be compared as if the model, split, and evaluation protocol are identical** to all Tasks 3 and 4 experiments.

## Conclusion
Dimensionality reduction substantially compresses the data, but the original representation is more predictive for both tasks under this experimental design. PCA may still be valuable for storage, visualization, or downstream runtime. LDA is especially compact for binary cancellation analysis.

## Outputs
`outputs/dimensionality_reduction/experiment_metrics.json`, `experiment_comparison.csv`, and two PCA explained-variance figures. Source: `src/dimensionality_reduction.py`.
