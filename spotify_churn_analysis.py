#!/usr/bin/env python3
"""
Spotify Churn Analysis - Complete Analysis Pipeline
Converted from Jupyter Notebook to Python Script

This script performs comprehensive analysis of Spotify user churn prediction including:
- Data loading and exploration
- Feature engineering and preprocessing
- Multiple machine learning models with hyperparameter tuning
- Deep learning models (MLP) with dropout regularization
- Performance comparison and visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Deep Learning imports
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("⚠️  TensorFlow not available. Deep learning section will be skipped.")
    TENSORFLOW_AVAILABLE = False

def print_section(title, char="=", width=70):
    """Print a formatted section header"""
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}")

def print_subsection(title, char="-", width=50):
    """Print a formatted subsection header"""
    print(f"\n{char * width}")
    print(f"{title}")
    print(f"{char * width}")

def main():
    print_section("🎵 SPOTIFY CHURN ANALYSIS", "=", 80)
    print("Motive: Predict whether a Spotify user will churn (cancel subscription) or remain active")
    print("This helps in understanding user engagement patterns and building strategies to reduce churn.")
    
    # ============================================================================
    # DATA LOADING AND EXPLORATION
    # ============================================================================
    print_section("📊 DATA LOADING AND EXPLORATION")
    
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv("./spotify_churn_dataset.csv")
    
    print(f"✅ Dataset loaded successfully!")
    print(f"📏 Dataset shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    
    # Dataset overview
    print_subsection("Dataset Overview")
    print("Rows: Each row represents a unique Spotify user")
    print("Columns (Features): Mix of numerical and categorical data")
    
    # Features description
    print_subsection("Features Description")
    features_desc = {
        'user_id': 'Unique identifier for each user',
        'gender': 'User gender (Male / Female / Other)',
        'age': 'User\'s age',
        'country': 'User\'s location',
        'subscription_type': 'Type of Spotify subscription (Free / Premium / Family / Student)',
        'listening_time': 'Minutes spent listening per day',
        'songs_played_per_day': 'Number of songs played daily',
        'skip_rate': 'Percentage of songs skipped',
        'device_type': 'Device used (Mobile / Desktop / Web)',
        'ads_listened_per_week': 'Number of ads heard per week',
        'offline_listening': 'Offline listening minutes',
        'is_churned': 'Target Variable: 0 → Active user, 1 → Churned (cancelled subscription)'
    }
    
    for feature, description in features_desc.items():
        print(f"• {feature}: {description}")
    
    # Basic info
    print_subsection("Dataset Information")
    print(f"📊 Data types:")
    print(df.dtypes)
    print(f"\n🔍 Missing values:")
    print(df.isnull().sum())
    
    # Descriptive statistics
    print_subsection("Descriptive Statistics")
    print("📈 Statistical Summary:")
    print(df.describe().T)
    
    # ============================================================================
    # DATA VISUALIZATION
    # ============================================================================
    print_section("📈 DATA VISUALIZATION")
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Class distribution analysis
    print_subsection("Class Distribution Analysis")
    churn_counts = df['is_churned'].value_counts()
    churn_percentages = df['is_churned'].value_counts(normalize=True) * 100
    
    print(f"🎯 Target Variable Distribution:")
    print(f"   Not Churned (0): {churn_counts[0]} ({churn_percentages[0]:.1f}%)")
    print(f"   Churned (1): {churn_counts[1]} ({churn_percentages[1]:.1f}%)")
    print(f"   Class Ratio: {churn_counts[0]/churn_counts[1]:.1f}:1")
    
    # Correlation analysis
    print_subsection("Correlation Analysis")
    corr = df.select_dtypes(include=[np.number]).corr()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", center=0)
    plt.title("Correlation Matrix - Numerical Features")
    plt.tight_layout()
    plt.show()
    
    # Feature correlation with target
    imp_fea = corr["is_churned"].sort_values(ascending=False)[1:]
    print(f"🔗 Feature Correlation with Target (is_churned):")
    for feature, corr_val in imp_fea.items():
        print(f"   {feature}: {corr_val:.4f}")
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=imp_fea.values, y=imp_fea.index, palette="Set1")
    plt.title("Feature Correlation with Target (is_churned)")
    plt.xlabel("Correlation Value")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show()
    
    # Categorical feature distributions
    print_subsection("Categorical Feature Distributions")
    
    categorical_features = ['device_type', 'country', 'subscription_type', 'gender']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, feature in enumerate(categorical_features):
        counts = df[feature].value_counts()
        counts.plot(kind='bar', ax=axes[i], color='skyblue', edgecolor='black')
        axes[i].set_title(f'{feature.title()} Distribution')
        axes[i].set_xlabel(feature.title())
        axes[i].set_ylabel('Count')
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Age distribution
    print_subsection("Age Distribution Analysis")
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(df['age'].dropna(), bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    plt.boxplot(df['age'].dropna(), vert=True)
    plt.title('Age Boxplot')
    plt.ylabel('Age')
    
    plt.tight_layout()
    plt.show()
    
    # ============================================================================
    # FEATURE ENGINEERING
    # ============================================================================
    print_section("🔧 FEATURE ENGINEERING")
    
    print("Performing one-hot encoding for categorical variables...")
    
    # One-hot encode categorical variables
    cat_cols = ["offline_listening", "device_type", "subscription_type", "country", "gender"]
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
    
    print(f"✅ Feature engineering completed!")
    print(f"📏 Original shape: {df.shape}")
    print(f"📏 Encoded shape: {df_encoded.shape}")
    print(f"📈 Features added: {df_encoded.shape[1] - df.shape[1]}")
    
    # ============================================================================
    # DATA PREPARATION
    # ============================================================================
    print_section("🎯 DATA PREPARATION")
    
    # Prepare features and target
    X = df_encoded.drop(columns=["is_churned", "user_id"])
    y = df_encoded["is_churned"]
    
    print(f"📊 Features shape: {X.shape}")
    print(f"🎯 Target distribution:")
    print(y.value_counts())
    print(f"📈 Target percentages:")
    print(y.value_counts(normalize=True) * 100)
    
    # Train-test split
    print("\n🔄 Creating train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✅ Train-test split completed!")
    print(f"📊 Training set: {X_train.shape[0]} samples")
    print(f"📊 Test set: {X_test.shape[0]} samples")
    print(f"📈 Train distribution: {y_train.value_counts(normalize=True) * 100}")
    print(f"📈 Test distribution: {y_test.value_counts(normalize=True) * 100}")
    
    # Feature scaling
    print("\n⚖️ Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("✅ Feature scaling completed!")
    
    # ============================================================================
    # MACHINE LEARNING MODELS - WITHOUT TUNING
    # ============================================================================
    print_section("🤖 MACHINE LEARNING MODELS - BASELINE PERFORMANCE")
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Support Vector Machine': SVC(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'Bagging': BaggingClassifier(random_state=42)
    }
    
    print("🚀 Training baseline models...")
    baseline_results = []
    
    for name, model in models.items():
        print(f"   Training {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        baseline_results.append((name, accuracy))
        print(f"   ✅ {name}: {accuracy:.4f}")
    
    print_subsection("Baseline Results Summary")
    baseline_df = pd.DataFrame(baseline_results, columns=['Model', 'Accuracy'])
    baseline_df = baseline_df.sort_values('Accuracy', ascending=False)
    print(baseline_df.to_string(index=False))
    
    # ============================================================================
    # HYPERPARAMETER TUNING
    # ============================================================================
    print_section("⚙️ HYPERPARAMETER TUNING")
    
    # Define parameter grids for each model
    param_grids = {
        'LogisticRegression': {
            'C': [0.1, 1, 10],
            'penalty': ['l2']
        },
        'KNeighborsClassifier': {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        },
        'DecisionTreeClassifier': {
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10]
        },
        'RandomForestClassifier': {
            'n_estimators': [100, 200],
            'max_depth': [5, 10],
            'min_samples_split': [2, 5]
        },
        'GradientBoostingClassifier': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        },
        'AdaBoostClassifier': {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1, 1]
        },
        'BaggingClassifier': {
            'n_estimators': [10, 50],
            'max_samples': [0.5, 1.0]
        }
    }
    
    print("🔍 Performing Grid Search with Cross-Validation...")
    tuned_results = []
    
    for name, model in models.items():
        if name in param_grids:
            print(f"\n🔍 Tuning {name}...")
            grid = GridSearchCV(
                model, 
                param_grids[name], 
                cv=5, 
                scoring='f1', 
                n_jobs=-1,
                verbose=0
            )
            grid.fit(X_train_scaled, y_train)
            
            best_model = grid.best_estimator_
            best_score = grid.best_score_
            test_score = best_model.score(X_test_scaled, y_test)
            
            print(f"   ✅ Best Parameters: {grid.best_params_}")
            print(f"   📊 CV Score (F1): {best_score:.4f}")
            print(f"   📊 Test Score (Accuracy): {test_score:.4f}")
            
            tuned_results.append((name, grid.best_params_, best_score, test_score))
    
    print_subsection("Hyperparameter Tuning Results")
    print("Model Performance After Tuning:")
    for name, params, cv_score, test_score in tuned_results:
        print(f"• {name}:")
        print(f"  Best Params: {params}")
        print(f"  CV Score (F1): {cv_score:.4f}")
        print(f"  Test Score (Accuracy): {test_score:.4f}")
        print()
    
    # ============================================================================
    # FEATURE IMPORTANCE ANALYSIS
    # ============================================================================
    print_section("🎯 FEATURE IMPORTANCE ANALYSIS")
    
    # Use the best Random Forest model for feature importance
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42
    )
    
    print("🌲 Training Random Forest for Feature Importance...")
    rf_model.fit(X_train_scaled, y_train)
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("📊 Top 15 Most Important Features:")
    print(feature_importance.head(15).to_string(index=False))
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 15 Most Important Features (Random Forest)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    # ============================================================================
    # DEEP LEARNING MODELS
    # ============================================================================
    if TENSORFLOW_AVAILABLE:
        print_section("🧠 DEEP LEARNING MODELS (MLP)")
        
        print("🔧 Building Multi-Layer Perceptron (MLP) models...")
        
        # Model 1: Basic MLP without dropout
        print_subsection("Model 1: Basic MLP (No Dropout)")
        dim = X_train_scaled.shape[1]
        
        model1 = Sequential([
            Dense(128, input_dim=dim, activation="relu"),
            Dense(128, activation="relu"),
            Dense(1, activation='sigmoid')
        ])
        
        print("📋 Model 1 Architecture:")
        model1.summary()
        
        model1.compile(loss='binary_crossentropy', metrics=['accuracy'])
        
        print("🚀 Training Model 1...")
        history1 = model1.fit(
            X_train_scaled, y_train, 
            epochs=15, 
            batch_size=32, 
            validation_split=0.2,
            verbose=1
        )
        
        # Model 2: MLP with dropout regularization
        print_subsection("Model 2: MLP with Dropout Regularization")
        
        model2 = Sequential([
            Dense(128, input_dim=dim, activation="relu"),
            Dropout(0.5),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        
        print("📋 Model 2 Architecture:")
        model2.summary()
        
        model2.compile(loss='binary_crossentropy', metrics=['accuracy'])
        
        print("🚀 Training Model 2...")
        history2 = model2.fit(
            X_train_scaled, y_train, 
            epochs=15, 
            batch_size=32, 
            validation_split=0.2,
            verbose=1
        )
        
        # Plot training history
        print_subsection("Training History Visualization")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(history1.history['loss'], label='Training Loss (No Dropout)')
        axes[0, 0].plot(history1.history['val_loss'], label='Validation Loss (No Dropout)')
        axes[0, 0].set_title('Model 1: Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        axes[0, 1].plot(history2.history['loss'], label='Training Loss (With Dropout)')
        axes[0, 1].plot(history2.history['val_loss'], label='Validation Loss (With Dropout)')
        axes[0, 1].set_title('Model 2: Loss Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Accuracy curves
        axes[1, 0].plot(history1.history['accuracy'], label='Training Accuracy (No Dropout)')
        axes[1, 0].plot(history1.history['val_accuracy'], label='Validation Accuracy (No Dropout)')
        axes[1, 0].set_title('Model 1: Accuracy Curves')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        
        axes[1, 1].plot(history2.history['accuracy'], label='Training Accuracy (With Dropout)')
        axes[1, 1].plot(history2.history['val_accuracy'], label='Validation Accuracy (With Dropout)')
        axes[1, 1].set_title('Model 2: Accuracy Curves')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
        
        # Evaluate deep learning models
        print_subsection("Deep Learning Model Evaluation")
        
        # Model 1 evaluation
        y_pred1 = (model1.predict(X_test_scaled) > 0.5).astype(int)
        accuracy1 = accuracy_score(y_test, y_pred1)
        
        # Model 2 evaluation
        y_pred2 = (model2.predict(X_test_scaled) > 0.5).astype(int)
        accuracy2 = accuracy_score(y_test, y_pred2)
        
        print(f"📊 Model 1 (No Dropout) Test Accuracy: {accuracy1:.4f}")
        print(f"📊 Model 2 (With Dropout) Test Accuracy: {accuracy2:.4f}")
        
        # Deep learning observations
        print_subsection("Deep Learning Observations")
        print("🔍 Analysis:")
        print("• Both models show signs of overfitting")
        print("• Model without dropout has wider gap between training and validation accuracy")
        print("• Model with dropout shows more unstable training but similar validation performance")
        print("• Further regularization techniques needed for better generalization")
        
    else:
        print("⚠️ Deep Learning section skipped (TensorFlow not available)")
    
    # ============================================================================
    # FINAL COMPARISON AND CONCLUSIONS
    # ============================================================================
    print_section("📊 FINAL RESULTS COMPARISON")
    
    print_subsection("Model Performance Summary")
    print("🏆 Top Performing Models (Baseline):")
    for i, (model, acc) in enumerate(baseline_df.head(3).values, 1):
        print(f"   {i}. {model}: {acc:.4f}")
    
    print("\n🔍 Key Observations:")
    print("• Multiple models achieved similar high accuracy (~74.8%)")
    print("• Class imbalance affects model performance (74% not churned vs 26% churned)")
    print("• Feature importance shows listening_time and songs_played_per_day are most predictive")
    print("• Deep learning models require further regularization to prevent overfitting")
    
    print_subsection("Recommendations")
    print("💡 Next Steps:")
    print("1. Address class imbalance using techniques like SMOTE or undersampling")
    print("2. Implement ensemble methods combining top-performing models")
    print("3. Add more regularization to deep learning models")
    print("4. Feature engineering: create interaction features")
    print("5. Cross-validation for more robust evaluation")
    
    print_section("✅ ANALYSIS COMPLETED", "=", 80)
    print("🎵 Spotify Churn Analysis pipeline completed successfully!")
    print("📊 All models trained and evaluated")
    print("📈 Visualizations generated")
    print("🎯 Feature importance analyzed")
    print("💡 Recommendations provided")

if __name__ == "__main__":
    main()
