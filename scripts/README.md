# Analysis Scripts Documentation

## Script Overview

### 1. `Telecom_Customer_Churn_Prediction_Insights.html`
**Purpose:** Complete end-to-end machine learning pipeline for telecom churn prediction  
**Format:** Jupyter Notebook exported as HTML for easy sharing  
**Size:** 601KB comprehensive analysis  
**Execution Time:** ~5-7 minutes for complete workflow  

#### Analysis Sections:
1. **Data Loading & Exploration:** Initial data inspection and quality checks
2. **Exploratory Data Analysis:** Statistical analysis and visualization
3. **Data Preprocessing:** Feature engineering and data cleaning
4. **Model Development:** ML algorithm comparison and selection
5. **Model Evaluation:** Performance metrics and validation
6. **Business Insights:** Interpretation and recommendations

## Detailed Analysis Workflow

### Section 1: Data Loading & Quality Assessment
```python
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
df = pd.read_csv('data/cleaned_churn.csv')

# Basic data exploration
print(f"Dataset Shape: {df.shape}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nBasic Statistics:\n{df.describe()}")

# Target variable distribution
print(f"\nChurn Distribution:\n{df['Churn'].value_counts(normalize=True)}")
```

### Section 2: Exploratory Data Analysis (EDA)

#### Churn Rate Analysis
```python
# Overall churn rate
churn_rate = df['Churn'].value_counts(normalize=True)['Yes'] * 100
print(f"Overall Churn Rate: {churn_rate:.2f}%")

# Churn by key demographics
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Churn by contract type
sns.countplot(data=df, x='Contract', hue='Churn', ax=axes[0,0])
axes[0,0].set_title('Churn by Contract Type')

# Churn by payment method
sns.countplot(data=df, x='PaymentMethod', hue='Churn', ax=axes[0,1])
axes[0,1].set_title('Churn by Payment Method')
axes[0,1].tick_params(axis='x', rotation=45)

# Churn by internet service
sns.countplot(data=df, x='InternetService', hue='Churn', ax=axes[1,0])
axes[1,0].set_title('Churn by Internet Service')

# Churn by tenure groups
df['TenureGroup'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72, 100],
                          labels=['0-12', '13-24', '25-48', '49-72', '72+'])
sns.countplot(data=df, x='TenureGroup', hue='Churn', ax=axes[1,1])
axes[1,1].set_title('Churn by Tenure Group')

plt.tight_layout()
plt.show()
```

#### Numerical Features Analysis
```python
# Distribution of numerical features
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Histograms
for i, feature in enumerate(numerical_features):
    sns.histplot(data=df, x=feature, hue='Churn', multiple='stack', ax=axes[0,i])
    axes[0,i].set_title(f'Distribution of {feature}')

# Box plots
for i, feature in enumerate(numerical_features):
    sns.boxplot(data=df, x='Churn', y=feature, ax=axes[1,i])
    axes[1,i].set_title(f'{feature} by Churn')

plt.tight_layout()
plt.show()

# Correlation analysis
numerical_df = df[numerical_features + ['Churn']].copy()
numerical_df['Churn'] = numerical_df['Churn'].map({'Yes': 1, 'No': 0})
correlation_matrix = numerical_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Numerical Features')
plt.show()
```

### Section 3: Data Preprocessing & Feature Engineering

#### Missing Value Handling
```python
# Handle missing TotalCharges
df['TotalCharges'] = df['TotalCharges'].replace(' ', '0')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(0, inplace=True)

# Verify no missing values remain
print(f"Missing values after cleaning:\n{df.isnull().sum()}")
```

#### Feature Engineering
```python
# Create derived features
df['AvgMonthlyCharge'] = df['TotalCharges'] / df['tenure']
df['AvgMonthlyCharge'].fillna(df['MonthlyCharges'], inplace=True)

# Service count
service_features = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 
                   'OnlineBackup', 'DeviceProtection', 'TechSupport',
                   'StreamingTV', 'StreamingMovies']
df['ServiceCount'] = df[service_features].apply(lambda x: (x == 'Yes').sum(), axis=1)

# Tenure-based features
df['NewCustomer'] = (df['tenure'] <= 12).astype(int)
df['LoyalCustomer'] = (df['tenure'] >= 48).astype(int)

print(f"Feature engineering completed. New shape: {df.shape}")
```

#### Categorical Encoding
```python
# Binary variable encoding
binary_mappings = {
    'gender': {'Male': 1, 'Female': 0},
    'Partner': {'Yes': 1, 'No': 0},
    'Dependents': {'Yes': 1, 'No': 0},
    'PhoneService': {'Yes': 1, 'No': 0},
    'PaperlessBilling': {'Yes': 1, 'No': 0}
}

for col, mapping in binary_mappings.items():
    df[col] = df[col].map(mapping)

# Target variable encoding
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# One-hot encoding for multi-category variables
categorical_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                   'OnlineBackup', 'DeviceProtection', 'TechSupport',
                   'StreamingTV', 'StreamingMovies', 'Contract', 
                   'PaymentMethod', 'TenureGroup']

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print(f"Final dataset shape after encoding: {df_encoded.shape}")
```

### Section 4: Model Development

#### Train-Test Split
```python
# Prepare features and target
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Churn rate in train: {y_train.mean():.3f}")
print(f"Churn rate in test: {y_test.mean():.3f}")
```

#### Model Training & Comparison
```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Initialize models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

# Train and evaluate models
results = {}
for name, model in models.items():
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'predictions': y_pred
    }
    
    print(f"{name} - Accuracy: {accuracy:.4f}")
```

### Section 5: Model Evaluation & Selection

#### Best Model Performance
```python
# Select best model
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['model']
best_predictions = results[best_model_name]['predictions']

print(f"\nBest Model: {best_model_name}")
print(f"Best Accuracy: {results[best_model_name]['accuracy']:.4f}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, best_predictions))

# Confusion Matrix
cm = confusion_matrix(y_test, best_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=['No Churn', 'Churn'],
           yticklabels=['No Churn', 'Churn'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
```

#### Feature Importance Analysis
```python
# Feature importance (for tree-based models)
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    # Top 15 features
    top_features = feature_importance.head(15)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_features, x='importance', y='feature')
    plt.title('Top 15 Feature Importance')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.show()
    
    print("Top 10 Most Important Features:")
    for i, row in top_features.head(10).iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")
```

### Section 6: Business Insights & Recommendations

#### Churn Risk Analysis
```python
# Predict churn probabilities for all customers
churn_probabilities = best_model.predict_proba(X)[:, 1]
df['ChurnProbability'] = churn_probabilities

# Risk segmentation
df['RiskSegment'] = pd.cut(df['ChurnProbability'], 
                          bins=[0, 0.3, 0.6, 0.8, 1.0],
                          labels=['Low', 'Medium', 'High', 'Very High'])

# Segment analysis
segment_analysis = df.groupby('RiskSegment').agg({
    'customerID': 'count',
    'MonthlyCharges': 'mean',
    'tenure': 'mean',
    'Churn': 'mean'
}).rename(columns={'customerID': 'CustomerCount'})

print("\nRisk Segment Analysis:")
print(segment_analysis)
```

#### Business Recommendations
```python
def generate_retention_recommendations(risk_segment, customer_data):
    recommendations = {
        'Very High': [
            "Immediate retention offer: 30% discount + premium service upgrade",
            "Personal retention call within 24 hours",
            "Free service upgrade for 3 months"
        ],
        'High': [
            "Targeted retention campaign: 20% discount + service bundle",
            "Email campaign within 48 hours",
            "Loyalty points bonus"
        ],
        'Medium': [
            "Proactive outreach: 10% discount + loyalty program",
            "Monthly newsletter with service tips",
            "Quarterly check-in calls"
        ],
        'Low': [
            "Standard customer service",
            "Monitor for service usage changes",
            "Annual satisfaction survey"
        ]
    }
    return recommendations.get(risk_segment, ["Standard service"])

# Generate recommendations for high-risk customers
high_risk_customers = df[df['RiskSegment'].isin(['High', 'Very High'])]
print(f"\nHigh-Risk Customers: {len(high_risk_customers)} ({len(high_risk_customers)/len(df)*100:.1f}%)")
print(f"Monthly Revenue at Risk: ${high_risk_customers['MonthlyCharges'].sum():,.0f}")

# Sample recommendations
sample_recommendations = generate_retention_recommendations('Very High', None)
print("\nSample Retention Recommendations for Very High Risk:")
for i, rec in enumerate(sample_recommendations, 1):
    print(f"{i}. {rec}")
```

## Advanced Analysis Extensions

### Customer Lifetime Value Analysis
```python
# Calculate CLV for different segments
df['CLV'] = df['MonthlyCharges'] * df['tenure'] * 12  # Simplified CLV

clv_by_segment = df.groupby('RiskSegment')['CLV'].agg(['mean', 'median', 'sum'])
print("\nCustomer Lifetime Value by Risk Segment:")
print(clv_by_segment)

# Visualize CLV distribution
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='RiskSegment', y='CLV')
plt.title('Customer Lifetime Value by Risk Segment')
plt.ylabel('CLV ($)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### Retention Strategy ROI Analysis
```python
def calculate_retention_roi(segment, discount_rate, retention_rate_increase):
    """
    Calculate ROI for retention strategies
    """
    segment_customers = df[df['RiskSegment'] == segment]
    monthly_revenue = segment_customers['MonthlyCharges'].sum()
    
    # Current churn loss
    current_churn_rate = segment_customers['Churn'].mean()
    monthly_churn_loss = monthly_revenue * current_churn_rate
    
    # After retention strategy
    new_churn_rate = current_churn_rate * (1 - retention_rate_increase)
    new_churn_loss = monthly_revenue * new_churn_rate
    
    # Monthly savings
    monthly_savings = monthly_churn_loss - new_churn_loss
    
    # Cost of retention (discounts)
    retention_cost = monthly_revenue * discount_rate
    
    # Net monthly benefit
    net_benefit = monthly_savings - retention_cost
    
    # ROI
    roi = (net_benefit / retention_cost) * 100 if retention_cost > 0 else 0
    
    return {
        'monthly_revenue': monthly_revenue,
        'monthly_savings': monthly_savings,
        'retention_cost': retention_cost,
        'net_benefit': net_benefit,
        'roi_percentage': roi
    }

# Calculate ROI for different strategies
strategies = [
    ('Very High', 0.30, 0.40),  # 30% discount, 40% retention improvement
    ('High', 0.20, 0.30),       # 20% discount, 30% retention improvement
    ('Medium', 0.10, 0.20)      # 10% discount, 20% retention improvement
]

print("\nRetention Strategy ROI Analysis:")
for segment, discount, retention_improvement in strategies:
    roi_analysis = calculate_retention_roi(segment, discount, retention_improvement)
    print(f"\n{segment} Risk Segment:")
    print(f"  Monthly Revenue: ${roi_analysis['monthly_revenue']:,.0f}")
    print(f"  Monthly Savings: ${roi_analysis['monthly_savings']:,.0f}")
    print(f"  Retention Cost: ${roi_analysis['retention_cost']:,.0f}")
    print(f"  Net Benefit: ${roi_analysis['net_benefit']:,.0f}")
    print(f"  ROI: {roi_analysis['roi_percentage']:.1f}%")
```

## Usage Instructions

### Running the Complete Analysis
```bash
# 1. Install required packages
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

# 2. Navigate to project directory
cd Telecom-Churn-Analysis

# 3. Open HTML analysis in browser
# Open scripts/Telecom_Customer_Churn_Prediction_Insights.html

# 4. For interactive analysis (if you have the .ipynb file)
# jupyter notebook scripts/Telecom_Customer_Churn_Prediction_Insights.ipynb
```

### Custom Analysis Template
```python
# Template for analyzing different customer segments
def analyze_customer_segment(df, segment_condition, segment_name):
    """
    Analyze specific customer segments
    """
    segment_data = df[segment_condition].copy()
    
    analysis = {
        'segment_size': len(segment_data),
        'churn_rate': segment_data['Churn'].mean(),
        'avg_monthly_charges': segment_data['MonthlyCharges'].mean(),
        'avg_tenure': segment_data['tenure'].mean(),
        'total_clv': segment_data['CLV'].sum()
    }
    
    print(f"\n{segment_name} Segment Analysis:")
    for key, value in analysis.items():
        print(f"  {key.replace('_', ' ').title()}: {value:.2f}")
    
    return segment_data, analysis

# Example usage
# fiber_customers, fiber_analysis = analyze_customer_segment(
#     df, df['InternetService_Fiber optic'] == 1, "Fiber Optic Customers"
# )
```

## Performance Optimization

### Memory Management
```python
# For large datasets, use memory-efficient processing
def memory_efficient_processing(file_path, chunk_size=1000):
    """
    Process large datasets in chunks
    """
    chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # Process each chunk
        processed_chunk = process_data_chunk(chunk)
        chunks.append(processed_chunk)
    
    # Combine results
    final_df = pd.concat(chunks, ignore_index=True)
    return final_df
```

### Parallel Processing
```python
# Parallel cross-validation for faster model training
from sklearn.model_selection import cross_val_score
from joblib import Parallel, delayed

def parallel_model_evaluation(model, X, y, cv=5):
    """
    Parallel model evaluation
    """
    def evaluate_fold(train_idx, test_idx):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_train, y_train)
        return model.score(X_test, y_test)
    
    # Generate CV splits
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Parallel execution
    scores = Parallel(n_jobs=-1)(
        delayed(evaluate_fold)(train_idx, test_idx)
        for train_idx, test_idx in skf.split(X, y)
    )
    
    return np.mean(scores), np.std(scores)
```

## Troubleshooting

### Common Issues
1. **Memory Errors:** Process data in chunks for large datasets
2. **Imbalanced Classes:** Use stratified sampling and class weights
3. **Feature Importance:** Not available for all model types
4. **Convergence Issues:** Increase max_iter for logistic regression

### Debug Tools
```python
# Model debugging utilities
def debug_model_performance(model, X_train, X_test, y_train, y_test):
    """
    Comprehensive model debugging
    """
    # Training performance
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"Training Score: {train_score:.4f}")
    print(f"Test Score: {test_score:.4f}")
    print(f"Overfitting Gap: {train_score - test_score:.4f}")
    
    # Prediction distribution
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    print(f"\nPrediction Probability Stats:")
    print(f"  Mean: {y_pred_proba.mean():.3f}")
    print(f"  Std: {y_pred_proba.std():.3f}")
    print(f"  Min: {y_pred_proba.min():.3f}")
    print(f"  Max: {y_pred_proba.max():.3f}")
    
    return train_score, test_score
```
