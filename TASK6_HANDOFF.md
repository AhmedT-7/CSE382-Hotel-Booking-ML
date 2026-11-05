# Task 6 handoff - CSE382 Hotel Booking ML

This package adds Task 6 without merging, resetting, or rewriting other team branches. The two reports include evidence from all five existing branches.

## Files to copy into the cloned repository

Copy the `src/`, `reports/`, `outputs/`, and `TASK6_HANDOFF.md` paths from the ZIP into the root of your clone, preserving their relative paths. The ZIP contains only Task 6 additions, not your dataset or `.git` directory.

## Run

From the project root, install the dependencies if necessary:

```
python -m pip install numpy pandas scikit-learn matplotlib
python src/dimensionality_reduction.py
```

The experiment will regenerate `outputs/dimensionality_reduction/experiment_metrics.json`, `experiment_comparison.csv`, and PCA variance plots. Runtime measurements may vary across computers. The supplied results were generated with scikit-learn 1.8.0 and random seed 42.

## Push safely

```
git switch feature/dimensionality-reduction
git status
git add src/dimensionality_reduction.py reports/task6_findings.md reports/complete_project_report.md reports/Task_6_Report.pdf reports/Complete_Project_Report.pdf outputs/dimensionality_reduction TASK6_HANDOFF.md
git commit -m "Add Task 6 PCA LDA experiments and project reports"
git push -u origin feature/dimensionality-reduction
```

Review `git status` before committing: Windows line endings may make the existing `data/H2.csv`, `README.md`, `.gitignore`, and `src/eda.py` appear modified even though Task 6 does not change them. Do not use `git add .` if those unintended modifications appear.

## Integration note

The original project stores Tasks 1-5 on separate branches. Task 6 has a self-contained implementation of the same feature policy and preprocessing strategy so that it runs directly from this branch based on `main`. For a future consolidated branch, refactor common logic into the team's `src/cleaning.py`, `src/features.py`, and `src/preprocessing.py` rather than maintaining duplicate implementations. No existing branch was changed by this work.
