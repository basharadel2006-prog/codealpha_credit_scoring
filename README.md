# Credit Scoring Model (Backend Only)

Objective: Predict creditworthiness using financial history (income, debt, late payments, credit utilization).\n
## Project Contents

- `credit_scoring.py`: synthetic dataset generation, preprocessing, model training (Logistic Regression, Decision Tree, Random Forest), evaluation, and best model export.
- `requirements.txt`: Python dependencies.
- `README.md`: instructions.

## Setup

1. Create and activate Python virtual environment.
   - `python -m venv venv`
   - `venv\\Scripts\\activate` (Windows)
2. Install dependencies:
   - `pip install -r requirements.txt`

## Run

- `python credit_scoring.py`

This will generate data, train models, print metrics (precision/recall/f1/roc_auc), and save:
- `best_credit_scoring_model.joblib`
- `credit_scoring_metrics.json`

## Notes

- Normal flow is backend-only (no frontend/GUI).
- You can adapt the synthetic generator to real data by loading a CSV and mapping feature columns.
- The metrics file and model are output in working directory.

## Optional GitHub / LinkedIn

- Commit these files and push to GitHub as the project source code.
- Record a short LinkedIn video describing problem, data generation, models compared, and final performance metrics.
