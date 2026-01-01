# 📊 Telecom Customer Churn Prediction & Insights

## 🔍 Overview
This project analyzes customer churn patterns in the telecom industry using Python and data visualization.  
The goal is to identify key drivers of churn and propose actionable business insights to improve customer retention.

## 🛠 Tools & Libraries
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Jupyter Notebook

## 📈 Key Metrics
- **Total Customers:** 7043  
- **Churn Rate:** 26.54%  
- **Average Tenure:** 32.37 months  
- **Average Monthly Charges:** $65  

## 📊 Dashboards
### Executive Summary
![Executive Summary]("D:\Telecom_Churn_Analysis\dashboards\executive_summary.png.png")

### Tenure vs Churn Analysis
![Tenure Analysis]("D:\Telecom_Churn_Analysis\dashboards\tenure_analysis.png")

### Payment & Revenue Risk
![Payment Risk]("D:\Telecom_Churn_Analysis\dashboards\executive_summary.png.png\payment_risk.png")

### High-Risk Segmentation
![High-Risk Segmentation]("D:\Telecom_Churn_Analysis\dashboards\executive_summary.png.png\high_risk_segmentation.png")

## 💡 Business Insights
- Month-to-month contracts have the highest churn rate (~43%).  
- Customers with **low tenure (<12 months)** are most likely to churn.  
- **Electronic check users** show higher churn risk compared to automatic payment methods.  
- Customers with **higher monthly charges** are more likely to churn, increasing revenue leakage risk.  

## 📁 Project Structure
Telecom-Churn-Analysis/
│
├── notebooks/
│   └── Telecom_Customer_Churn_Prediction_Insights.ipynb
├── data/
│   └── cleaned_churn.csv
├── dashboards/
│   ├── executive_summary.png
│   ├── tenure_analysis.png
│   ├── payment_risk.png
│   └── high_risk_segmentation.png
├── README.md
└── requirements.txt


## ▶️ How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/harshmeena9977-ops/Telecom-Churn-Analysis

pip install -r requirements.txt

jupyter notebook notebooks/Telecom_Customer_Churn_Prediction_Insights.ipynb

