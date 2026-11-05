# CSE382 Hotel Booking ML

End-to-end machine learning project for CSE382 using the City Hotel (H2) Booking Demand dataset: EDA and cleaning, ADR regression, cancellation classification, booking segmentation with clustering, and PCA/LDA experiments.

- **GitHub repository:** https://github.com/AhmedT-7/CSE382-Hotel-Booking-ML
- **Submitted commit (Milestone 1):** 976a2308f3f72eb02df11e8d2d5460de165f360d

## Team

| Member | Name | GitHub account | Task |
|---|---|---|---|
| 1 | Ahmed Tamer Shahin | AhmedT-7 | EDA and data cleaning |
| 2 | Aley Derbala | aleyderbala-alt | Preprocessing and feature engineering |
| 3 | Mohamed Fouad | mohamedahmedfouad007 | Regression (ADR) |
| 4 | Marwan Khattab | marwankhattab | Classification (IsCanceled) |
| 5 | Ali Amr | ali4mr | Clustering |
| 6 | Aley Abdelkader | aleyabdelkader | Dimensionality reduction (PCA/LDA) and integration |

## Setup

Requires Python 3.10 or newer.

```bash
git clone https://github.com/AhmedT-7/CSE382-Hotel-Booking-ML.git
cd CSE382-Hotel-Booking-ML
python -m venv .venv
```

Activate the environment:

- Windows (PowerShell): `.venv\Scripts\Activate.ps1`
- Windows (Git Bash): `source .venv/Scripts/activate`
- macOS / Linux: `source .venv/bin/activate`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

Run all commands from the project root.

| Step | Command | Output |
|---|---|---|
| EDA | `python src/eda.py` | Dataset summary and plots |
| Cleaning check | `python src/cleaning.py` | Cleaning summary |
| Feature sets | `python src/features.py` | Task-specific feature lists |
| Preprocessing | `python src/preprocessing.py` | Train/validation/test shapes |
| Regression | `python src/regression.py` | ADR model results; models saved to `artifacts/` |
| Classification baseline | `python src/classification.py` | Cancellation model results |
| Classification tuning | `python src/tune_classification.py` | Tuned cancellation results (several minutes) |
| Clustering | `python src/clustering.py` | K-Means metrics, profile and PCA plot in `outputs/clustering/` |
| PCA / LDA | `python src/dimensionality_reduction.py` | Results in `outputs/dimensionality_reduction/` |

`src/eda.py` opens plot windows; close each one to continue.

## Project structure

```
data/        H2.csv dataset
src/         source code, one module per task
outputs/     generated result tables and figures
reports/     progress report and task reports
artifacts/   saved models (generated locally, not committed)
```

## Report

The Milestone 1 progress report is `reports/Complete_Project_Report.pdf`.
