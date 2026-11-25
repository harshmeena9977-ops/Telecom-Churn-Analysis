# Telecom Customer Churn Analysis

An independent practice project exploring customer churn patterns with a saved Jupyter HTML export, a tracked cleaned-data CSV, a Power BI report, and chart exports. The repository does not contain a trained prediction model, so this project is presented as exploratory analysis rather than churn prediction.

## Repository contents

- [Saved analysis (HTML export)](scripts/Telecom%20Customer%20Churn%20Prediction%20%26%20Insights.html)
- [Tracked cleaned-data CSV](data/cleaned_churn.csv.csv)
- [Power BI report](outputs/Telco%20Customer%20Churn%20Dashboard.pbix)
- [Executive summary chart](outputs/executive_summary.png.png)
- [High-risk segmentation chart](outputs/high_risk_segmentation.png.png)
- [Payment analysis chart](outputs/payment_risk.png.png)
- [Tenure analysis chart](outputs/tenure_analysis.png.png)
- [Machine-learning pipeline notes](docs/machine_learning_pipeline.md)
- [Dependency notes](docs/requirment.txt.txt)

## What the saved analysis contains

The HTML export includes data cleaning and exploratory charts for churn by tenure, contract, internet service, monthly charges, and payment method. It also contains an independent-samples t-test comparing tenure for churned and retained customers.

The checked-in analysis is an HTML export, not an executable notebook. Its saved outputs have not been reproduced from a clean environment. The t-test describes a tenure difference in this dataset; it does not establish why customers churn or prove that any retention action will work.

## Data and reproducibility limits

- The HTML export loads its source CSV from a personal Windows Downloads path. That raw file is not included in this repository, and the README does not have a dataset citation or license information.
- The tracked data file is named `data/cleaned_churn.csv.csv`. The HTML code writes `data/cleaned_churn.csv`, so the tracked filename and notebook output path do not match.
- The repository has no `.ipynb` source notebook or root-level `requirements.txt`. The dependency note is a plain text file at `docs/requirment.txt.txt`.
- The README previously described model training, accuracy, precision, recall, F1, feature importance, revenue at risk, and business impact. Those claims are not present in the checked-in HTML analysis and are not made here.

## Power BI and chart files

The repository includes a PBIX file and four PNG exports. They are linked above; their underlying measures and figures have not been independently reconciled with the CSV in this README update.

## Project status

Exploratory practice analysis. Dataset provenance, executable notebook source, dependency setup, file naming, KPI reconciliation, and reproducible analysis need attention before stronger conclusions or predictive claims can be made.
