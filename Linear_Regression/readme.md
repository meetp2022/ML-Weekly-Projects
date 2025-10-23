# Week 1 — Linear Regression (Housing prices)

## Project goal
Implement a linear regression pipeline on a housing dataset (California Housing or equivalent). Perform EDA, preprocessing, model training, evaluation, and a short improvement experiment.

## Files
- `notebooks/linear_regression_starter.ipynb` — the main notebook with EDA, training & evaluation.
- `data/` — (small sample or data link)
- `src/` — helper scripts (optional)
- `results/` — plots and saved model artifacts
- `requirements.txt` — packages

## How to run
1. Create virtual environment: `python -m venv .venv`
2. Activate it:
   - macOS/Linux: `source .venv/bin/activate`
   - Windows (PowerShell): `.venv\Scripts\Activate.ps1`
3. Install requirements: `pip install -r requirements.txt`
4. Open the notebook in VS Code or run `jupyter notebook`.

## Summary
Executive summary

- Dataset: California Housing (scikit-learn). Features include demographics and geographic predictors (e.g., MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude). Target is median house value per block group (MedHouseVal).
- Model: Ordinary least squares LinearRegression trained on 80% of the data (random_state=42).
- Key metrics: RMSE and R² on the held-out 20% test set. (The notebook prints exact values; paste them here if you want numeric clarity — e.g., "RMSE: 0.XXX, R²: 0.XXX".)
- A simple linear model explains a meaningful portion of variance (R² > 0 indicates predictive power), but residual error remains (non-zero RMSE). Visual diagnostics suggest the model captures the central trend but deviates on extremes.
