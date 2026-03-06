# Machine Learning Pipeline Documentation

## Model Development Framework

### 1. Data Preprocessing Pipeline

#### Missing Value Handling
```python
# Strategy for missing values
def handle_missing_values(df):
    # TotalChains has missing values for new customers
    df['TotalCharges'] = df['TotalCharges'].replace(' ', '0')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)
    
    # Fill missing categorical modes
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df
```

#### Feature Engineering
```python
# Create derived features
def feature_engineering(df):
    # Tenure groups
    df['TenureGroup'] = pd.cut(df['tenure'], 
                              bins=[0, 12, 24, 48, 72, 100],
                              labels=['0-12', '13-24', '25-48', '49-72', '72+'])
    
    # Service count
    service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 
                   'OnlineBackup', 'DeviceProtection', 'TechSupport',
                   'StreamingTV', 'StreamingMovies']
    df['ServiceCount'] = df[service_cols].apply(lambda x: (x == 'Yes').sum(), axis=1)
    
    # Average monthly charge
    df['AvgMonthlyCharge'] = df['TotalCharges'] / df['tenure']
    df['AvgMonthlyCharge'].fillna(df['MonthlyCharges'], inplace=True)
    
    return df
```

#### Categorical Encoding
```python
# Label encoding for binary variables
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})

# One-hot encoding for categorical variables
categorical_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                   'OnlineBackup', 'DeviceProtection', 'TechSupport',
                   'StreamingTV', 'StreamingMovies', 'Contract', 
                   'PaymentMethod', 'TenureGroup']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Target variable encoding
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
```

### 2. Model Selection & Training

#### Algorithm Comparison
```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'SVM': SVC(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

# Cross-validation comparison
results = {}
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    results[name] = cv_scores.mean()
    print(f"{name}: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
```

#### Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

# Random Forest hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_
```

### 3. Model Evaluation Metrics

#### Classification Metrics
```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Detailed classification report
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Confusion matrix visualization
import seaborn as sns
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=['No Churn', 'Churn'], 
           yticklabels=['No Churn', 'Churn'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# ROC AUC Score
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC Score: {roc_auc:.4f}")
```

#### Feature Importance Analysis
```python
# Feature importance for Random Forest
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

# Top 10 features
top_features = feature_importance.head(10)
plt.figure(figsize=(10, 6))
sns.barplot(data=top_features, x='importance', y='feature')
plt.title('Top 10 Feature Importance')
plt.xlabel('Importance Score')
plt.show()
```

### 4. Advanced Modeling Techniques

#### Ensemble Methods
```python
# Voting Classifier
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingClassifier(random_state=42)),
        ('lr', LogisticRegression(random_state=42, max_iter=1000))
    ],
    voting='soft'
)

voting_clf.fit(X_train, y_train)
voting_score = voting_clf.score(X_test, y_test)
```

#### Stacking Ensemble
```python
from sklearn.ensemble import StackingClassifier

stacking_clf = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingClassifier(random_state=42))
    ],
    final_estimator=LogisticRegression(random_state=42, max_iter=1000)
)

stacking_clf.fit(X_train, y_train)
stacking_score = stacking_clf.score(X_test, y_test)
```

### 5. Model Interpretation

#### SHAP Values for Explainability
```python
import shap

# SHAP analysis for Random Forest
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values[1], X_test, plot_type="bar")

# Individual prediction explanation
shap.force_plot(explainer.expected_value[1], shap_values[1][0], X_test.iloc[0])
```

#### Partial Dependence Plots
```python
from sklearn.inspection import plot_partial_dependence

# Partial dependence for top features
features = ['tenure', 'MonthlyCharges', 'Contract_Month-to-month']
plot_partial_dependence(best_model, X_test, features, 
                       feature_names=X_test.columns)
plt.show()
```

### 6. Business Application Framework

#### Churn Risk Scoring
```python
def calculate_churn_risk(customer_data, model, feature_columns):
    """
    Calculate churn risk score for individual customers
    """
    # Preprocess customer data
    customer_processed = preprocess_customer_data(customer_data)
    
    # Ensure correct feature order
    customer_features = customer_processed[feature_columns]
    
    # Predict churn probability
    churn_probability = model.predict_proba(customer_features)[:, 1][0]
    
    # Risk categorization
    if churn_probability >= 0.8:
        risk_level = "Very High"
    elif churn_probability >= 0.6:
        risk_level = "High"
    elif churn_probability >= 0.4:
        risk_level = "Medium"
    else:
        risk_level = "Low"
    
    return {
        'churn_probability': churn_probability,
        'risk_level': risk_level,
        'recommendation': get_retention_recommendation(risk_level, customer_data)
    }

def get_retention_recommendation(risk_level, customer_data):
    recommendations = {
        "Very High": "Immediate retention offer - 30% discount + premium service upgrade",
        "High": "Targeted retention campaign - 20% discount + service bundle",
        "Medium": "Proactive outreach - 10% discount + loyalty program",
        "Low": "Standard customer service - monitor for changes"
    }
    return recommendations[risk_level]
```

#### Customer Segmentation
```python
def segment_customers(df, model):
    """
    Segment customers based on churn risk and value
    """
    # Predict churn probabilities
    df['churn_probability'] = model.predict_proba(df)[:, 1]
    
    # Calculate customer lifetime value
    df['clv'] = df['MonthlyCharges'] * df['tenure'] * 12  # Simplified CLV
    
    # Create segments
    conditions = [
        (df['churn_probability'] >= 0.7) & (df['clv'] >= df['clv'].quantile(0.75)),
        (df['churn_probability'] >= 0.7) & (df['clv'] < df['clv'].quantile(0.75)),
        (df['churn_probability'] < 0.7) & (df['clv'] >= df['clv'].quantile(0.75)),
        (df['churn_probability'] < 0.7) & (df['clv'] < df['clv'].quantile(0.75))
    ]
    
    choices = ['High-Risk High-Value', 'High-Risk Low-Value', 
              'Low-Risk High-Value', 'Low-Risk Low-Value']
    
    df['segment'] = np.select(conditions, choices)
    
    return df
```

### 7. Model Monitoring & Maintenance

#### Performance Tracking
```python
def model_performance_monitoring(y_true, y_pred, y_pred_proba, timestamp):
    """
    Track model performance over time
    """
    metrics = {
        'timestamp': timestamp,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_pred_proba)
    }
    
    # Store metrics in database or log file
    log_model_performance(metrics)
    
    return metrics

def detect_model_drift(current_metrics, baseline_metrics, threshold=0.05):
    """
    Detect if model performance is degrading
    """
    drift_detected = False
    
    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        current_value = current_metrics[metric]
        baseline_value = baseline_metrics[metric]
        
        if abs(current_value - baseline_value) > threshold:
            drift_detected = True
            print(f"Drift detected in {metric}: {current_value:.4f} vs {baseline_value:.4f}")
    
    return drift_detected
```

#### Retraining Strategy
```python
def automated_retraining_pipeline(new_data, model, performance_threshold=0.8):
    """
    Automated model retraining when performance degrades
    """
    # Evaluate current model on new data
    current_performance = evaluate_model_performance(model, new_data)
    
    if current_performance['accuracy'] < performance_threshold:
        print("Performance below threshold - initiating retraining")
        
        # Retrain model with new data
        X_new = new_data.drop('Churn', axis=1)
        y_new = new_data['Churn']
        
        # Retrain with hyperparameter tuning
        best_model = hyperparameter_tuning(X_new, y_new)
        
        # Validate new model
        new_performance = evaluate_model_performance(best_model, new_data)
        
        if new_performance['accuracy'] > current_performance['accuracy']:
            print("Retraining successful - model updated")
            return best_model
        else:
            print("Retraining unsuccessful - keeping current model")
            return model
    else:
        print("Model performance acceptable - no retraining needed")
        return model
```

## Quality Assurance & Validation

### Cross-Validation Strategy
```python
from sklearn.model_selection import StratifiedKFold

# Stratified cross-validation for imbalanced dataset
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = []
for train_idx, val_idx in skf.split(X, y):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    
    # Train and evaluate
    model.fit(X_train_fold, y_train_fold)
    fold_score = model.score(X_val_fold, y_val_fold)
    cv_results.append(fold_score)

print(f"CV Score: {np.mean(cv_results):.4f} (+/- {np.std(cv_results):.4f})")
```

### Robustness Testing
```python
# Test model on different data subsets
def robustness_test(model, X, y, sample_sizes=[0.5, 0.7, 0.9]):
    results = {}
    
    for size in sample_sizes:
        # Sample data
        X_sample, _, y_sample, _ = train_test_split(X, y, train_size=size, random_state=42)
        
        # Evaluate
        score = model.score(X_sample, y_sample)
        results[f'{int(size*100)}%'] = score
    
    return results
```
