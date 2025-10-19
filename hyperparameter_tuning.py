import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('/home/bekzhan/Code/spotify-dm/spotify_churn_dataset.csv')

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Data preprocessing
print("\n=== Data Preprocessing ===")

# Check for missing values
print("Missing values:")
print(df.isnull().sum())

# One-hot encode categorical variables
cat_cols = ['gender', 'country', 'subscription_type', 'device_type']
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

print(f"\nDataset shape after encoding: {df_encoded.shape}")

# Prepare features and target
X = df_encoded.drop('is_churned', axis=1)
y = df_encoded['is_churned']

print(f"Features shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Scale features for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n=== HYPERPARAMETER TUNING ===")

# 1. Logistic Regression Hyperparameter Tuning
print("\n1. LOGISTIC REGRESSION HYPERPARAMETER TUNING")
print("=" * 50)

# Reduced parameter grid for faster tuning
lr_param_grid_filtered = {
    'C': [0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear'],
    'max_iter': [1000],
    'class_weight': ['balanced', None]
}

lr_grid_search = GridSearchCV(
    LogisticRegression(random_state=42),
    lr_param_grid_filtered,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

print("Training Logistic Regression with GridSearchCV...")
lr_grid_search.fit(X_train_scaled, y_train)

print(f"Best parameters: {lr_grid_search.best_params_}")
print(f"Best cross-validation score: {lr_grid_search.best_score_:.4f}")

# 2. Decision Tree Hyperparameter Tuning
print("\n2. DECISION TREE HYPERPARAMETER TUNING")
print("=" * 50)

# Reduced parameter grid for faster tuning
dt_param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', None],
    'class_weight': ['balanced', None]
}

dt_grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    dt_param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

print("Training Decision Tree with GridSearchCV...")
dt_grid_search.fit(X_train, y_train)

print(f"Best parameters: {dt_grid_search.best_params_}")
print(f"Best cross-validation score: {dt_grid_search.best_score_:.4f}")

# 3. Random Forest Hyperparameter Tuning
print("\n3. RANDOM FOREST HYPERPARAMETER TUNING")
print("=" * 50)

# Reduced parameter grid for faster tuning
rf_param_grid = {
    'n_estimators': [100, 200],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', None],
    'bootstrap': [True],
    'class_weight': ['balanced', None]
}

rf_grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

print("Training Random Forest with GridSearchCV...")
rf_grid_search.fit(X_train, y_train)

print(f"Best parameters: {rf_grid_search.best_params_}")
print(f"Best cross-validation score: {rf_grid_search.best_score_:.4f}")

# 4. RandomizedSearchCV for additional exploration (reduced iterations)
print("\n4. RANDOMIZED SEARCH FOR ADDITIONAL EXPLORATION")
print("=" * 50)

# Logistic Regression RandomizedSearch (reduced iterations)
lr_random_params = {
    'C': np.logspace(-2, 2, 10),
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear'],
    'max_iter': [1000, 2000],
    'class_weight': ['balanced', None]
}

lr_random_search = RandomizedSearchCV(
    LogisticRegression(random_state=42),
    lr_random_params,
    n_iter=20,  # Reduced from 50
    cv=3,      # Reduced from 5
    scoring='f1',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

print("Training Logistic Regression with RandomizedSearchCV...")
lr_random_search.fit(X_train_scaled, y_train)

print(f"Best parameters: {lr_random_search.best_params_}")
print(f"Best cross-validation score: {lr_random_search.best_score_:.4f}")

# Decision Tree RandomizedSearch (reduced iterations)
dt_random_params = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': range(2, 11),
    'min_samples_leaf': range(1, 6),
    'max_features': ['sqrt', 'log2', None],
    'class_weight': ['balanced', None]
}

dt_random_search = RandomizedSearchCV(
    DecisionTreeClassifier(random_state=42),
    dt_random_params,
    n_iter=20,  # Reduced from 50
    cv=3,       # Reduced from 5
    scoring='f1',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

print("Training Decision Tree with RandomizedSearchCV...")
dt_random_search.fit(X_train, y_train)

print(f"Best parameters: {dt_random_search.best_params_}")
print(f"Best cross-validation score: {dt_random_search.best_score_:.4f}")

# Random Forest RandomizedSearch (reduced iterations)
rf_random_params = {
    'n_estimators': range(50, 300, 50),
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': range(2, 11),
    'min_samples_leaf': range(1, 6),
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False],
    'class_weight': ['balanced', None]
}

rf_random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    rf_random_params,
    n_iter=20,  # Reduced from 50
    cv=3,       # Reduced from 5
    scoring='f1',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

print("Training Random Forest with RandomizedSearchCV...")
rf_random_search.fit(X_train, y_train)

print(f"Best parameters: {rf_random_search.best_params_}")
print(f"Best cross-validation score: {rf_random_search.best_score_:.4f}")

# 5. Model Evaluation with Best Parameters
print("\n5. MODEL EVALUATION WITH BEST PARAMETERS")
print("=" * 50)

# Get best models
best_lr_grid = lr_grid_search.best_estimator_
best_dt_grid = dt_grid_search.best_estimator_
best_rf_grid = rf_grid_search.best_estimator_

best_lr_random = lr_random_search.best_estimator_
best_dt_random = dt_random_search.best_estimator_
best_rf_random = rf_random_search.best_estimator_

# Evaluate models
models = {
    'LR_Grid': best_lr_grid,
    'DT_Grid': best_dt_grid,
    'RF_Grid': best_rf_grid,
    'LR_Random': best_lr_random,
    'DT_Random': best_dt_random,
    'RF_Random': best_rf_random
}

results = []

for name, model in models.items():
    # Make predictions
    if 'LR' in name:
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    results.append([name, acc, prec, rec, f1, auc])
    
    print(f"\n{name} Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Classification Report:")
    print(classification_report(y_test, y_pred))

# Create results DataFrame
results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'])
print("\n=== FINAL RESULTS COMPARISON ===")
print(results_df.round(4))

# 6. Feature Importance Analysis
print("\n6. FEATURE IMPORTANCE ANALYSIS")
print("=" * 40)

# Get feature importance from Random Forest
best_rf = best_rf_random if best_rf_random.score(X_test, y_test) > best_rf_grid.score(X_test, y_test) else best_rf_grid

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 15 Most Important Features:")
print(feature_importance.head(15))

# Plot feature importance
plt.figure(figsize=(10, 8))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance')
plt.title('Top 15 Most Important Features (Random Forest)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# 7. Confusion Matrices
print("\n7. CONFUSION MATRICES")
print("=" * 30)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i, (name, model) in enumerate(models.items()):
    if 'LR' in name:
        y_pred = model.predict(X_test_scaled)
    else:
        y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
    axes[i].set_title(f'{name} Confusion Matrix')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# 8. ROC Curves
print("\n8. ROC CURVES COMPARISON")
print("=" * 30)

plt.figure(figsize=(10, 8))

for name, model in models.items():
    if 'LR' in name:
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_prob = model.predict_proba(X_test)[:, 1]
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)
    
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend()
plt.grid(True)
plt.show()

print("\n=== HYPERPARAMETER TUNING COMPLETED ===")
print("Best performing models based on F1-Score:")
best_models = results_df.nlargest(3, 'F1-Score')
print(best_models[['Model', 'F1-Score', 'AUC']])
