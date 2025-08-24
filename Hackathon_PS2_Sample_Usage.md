# Multivariate Time Series Anomaly Detection Solution

## Overview

This is a self-contained anomaly detection pipeline designed for time-series sensor data. It satisfies all functional, technical, and evaluation criteria outlined in the Hackathon Problem Statement-2.

The solution detects anomalies using a hybrid approach (Isolation Forest + PCA) and explains each anomaly with per-row top contributing features (Explainable AI).

---

## Sample Usage

To run the code, use the following command:

```
python main.py --input MVTA_data.csv --output anomaly_results_final.csv --plots ./plots_final
```

### Parameters:

- `--input`: Path to input CSV file (first column must be timestamp)
- `--output`: Path where output CSV will be saved (includes 8 extra columns)
- `--plots`: Folder path to save all visualizations as `.png` images

---

## Output Files

1. `anomaly_results_final.csv`

   - Contains original data
   - `anomaly_score` (calibrated 0–100)
   - `top_feature_1` to `top_feature_7`

2. `./plots_final/`

   - Time-series score plots
   - Histogram of scores
   - Rolling mean plots
   - Attribution frequency and heatmaps

3. Console output includes:
   - Training period validation
   - Sudden jump count
   - Hackathon criteria check

---

## Requirements

Create a virtual environment and install dependencies:

```
python -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### requirements.txt

```
pandas
numpy
scikit-learn
matplotlib
```
