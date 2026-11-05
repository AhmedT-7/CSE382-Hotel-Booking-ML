# CSE382 Hotel Booking Machine Learning - Complete Project Report

## Executive summary
This project analyzes the City Hotel (H2) booking demand dataset and addresses data understanding, cleaning, supervised ADR prediction, booking cancellation prediction, customer segmentation, and dimensionality reduction. Tasks 1-5 are recorded in separate repository branches; Task 6 is added as a standalone module. Because the branches are not all integrated into `main`, this report synthesizes branch-specific evidence rather than claiming that one combined end-to-end executable already exists.

## Dataset and study scope
The H2 data contains 79,330 rows and 31 columns (Task 1 findings). `ADR` is the regression target and `IsCanceled` the classification target. Approximately 41.73% of bookings were canceled. The study focuses on predictors available at booking creation to avoid information leakage from later outcomes.

## Task 1 - Exploratory Data Analysis and Data Cleaning
Branch: `feature/eda`. The team measured four missing Children values, 24 missing Country values, 8,131 literal NULL Agent values, and 75,641 literal NULL Company values. They identified 25,902 duplicate records and 167 zero-guest records. Outlier counts using a 1.5-IQR rule included 3,387 ADR and 2,576 LeadTime values. EDA charts illustrated right-skewed ADR and lead times, cancellation distribution, and their relationships. Cleaning strips whitespace, fills Children with zero, normalizes selected NULLs, and optionally removes zero-guest records; original rows and outliers are otherwise retained.

## Task 2 - Preprocessing and Feature Engineering
Branch: `feature/preprocessing`. The team built booking-time feature lists for classification and regression, excluded post-booking information, and engineered TotalNights and TotalGuests. Numerical columns receive median imputation and standardization; categorical columns receive constant imputation and one-hot encoding. GroupShuffleSplit divides data into approximately 70% train, 15% validation, and 15% test while keeping hashed duplicate rows grouped. Preprocessing must be fit on training data only.

## Task 3 - Regression and Optimization
Branch: `Regression`. The regression task predicts ADR. The implementation compares Linear Regression, Decision Tree Regression, and Random Forest Regression. Tree and forest variants undergo GridSearchCV; metrics include MAE, RMSE, and R2. Models and the preprocessing object are serialized with joblib. The inspected branch contains source code, but no committed measured regression score table; numeric performance claims are therefore not made for Task 3.

## Task 4 - Classification and Optimization
Branch: `feature/classification`. The task predicts booking cancellation with Logistic Regression, Random Forest, and RBF SVM. The baseline report records Logistic Regression validation accuracy 0.7724 and ROC-AUC 0.8199; Random Forest 0.7679 and 0.8311; RBF SVM 0.7748 and 0.8146 (SVM used 5,000 training rows). In the tuning report, Random Forest leads the compared tuned methods: validation accuracy 0.7759, ROC-AUC 0.8529, cancellation F1 0.72. Logistic Regression achieved accuracy 0.7684 and ROC-AUC 0.8202; tuned SVM achieved 0.7591 and 0.8200. Tuning uses validation metrics and grouped cross-validation; final test-set evaluation was not reported.

## Task 5 - Clustering and Optimization
Branch: `clustering`. The baseline uses engineered numeric booking attributes and categories, with log-transformation of selected skewed features, scaling, and one-hot encoding. K-Means uses k=4 on 79,163 cleaned bookings, producing 32 encoded features. Its committed baseline metrics show silhouette 0.16406 and Davies-Bouldin 1.68161; an environment-specific fit time of 0.75 seconds is recorded. The branch includes a PCA 2D cluster plot and a cluster profile CSV. These values describe the baseline, not a claim of globally optimal k.

## Task 6 - Dimensionality Reduction and ML Experiments
Branch: `feature/dimensionality-reduction` (prepared handoff). The new script tests PCA retaining 95% variance and LDA (classification only), using train-fitted pipelines. For cancellation prediction, the original 58-dimensional representation achieved validation ROC-AUC 0.79891; PCA reduced 58 to 21 dimensions with ROC-AUC 0.78855; LDA reduced to one dimension with ROC-AUC 0.78885. For ADR, the original 213-dimensional representation achieved RMSE 23.19222 and R2 0.65124; PCA reduced to 28 dimensions with RMSE 28.19475 and R2 0.48456. See the dedicated Task 6 report for full protocol and all metrics.

## Integration, caveats, and next steps
The branches use related preprocessing ideas but are not yet a fully merged codebase. The regression branch evaluates baseline and tuned models on the test set during its model loop; future comparisons should instead reserve test data until final model selection. Tasks 4 and 6 use validation reporting and keep the test set unexamined. Grouped random splits mitigate identical-row leakage but do not establish temporal generalization. Future integration should unify the feature policy, record environment versions, use one shared split artifact, and perform a single held-out final test comparison.

## Overall conclusion
The team has built complementary analytical components: EDA, structured preprocessing, supervised models, clustering, and dimensionality reduction. Tuned Random Forest is currently strongest among the documented Task 4 classifiers. Task 6 demonstrates that aggressive compression trades away validation accuracy for compact representations in these baseline models. These findings motivate further tuning, integration, and rigorous final testing rather than a universal recommendation for or against reduction.

## Evidence sources
Repository: https://github.com/AhmedT-7/CSE382-Hotel-Booking-ML . Branch artifacts: `feature/eda:reports/eda_findings.md`, `feature/classification:reports/classification_baseline.md`, `feature/classification:reports/classification_tuning.md`, `clustering:outputs/clustering/kmeans_baseline_metrics.json`, and corresponding `src/` modules; Task 6 generated outputs and source in the handoff package.
