import pandas as pd
import numpy as np
import os
import joblib
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

# Create output directories
os.makedirs('reports/visualizations', exist_ok=True)
os.makedirs('models', exist_ok=True)

def perform_feature_engineering(df):
    """
    Advanced Feature Engineering as per requirements:
    discount_percent, price_per_rating, delivery_cost, store_encoded, demand_score, popularity_score, price_trend, book_age, seasonal_factor
    """
    df = df.copy()
    
    # 1. Feature: discount_percent (If not existing, generate mock or use existing)
    if 'discount_percent' not in df.columns:
        df['discount_percent'] = np.random.uniform(5, 45, len(df))
    
    # 2. Feature: price_per_rating
    # Handle cases where rating is zero
    df['price_per_rating'] = df['price'] / (df['rating'].replace(0, 0.1))
    
    # 3. Feature: delivery_cost (Existing or mock)
    if 'delivery_cost' not in df.columns:
        df['delivery_cost'] = np.random.choice([0, 40, 50, 75], size=len(df))
    
    # 4. Feature: demand_score
    df['demand_score'] = (df['reviews_count'] / (df['reviews_count'].max() + 1)) * 100
    
    # 5. Feature: popularity_score
    df['popularity_score'] = (df['rating'] * 15) + (df['reviews_count'] / 2500)
    
    # 6. Feature: price_trend (Mocking based on years if not available)
    # Using year to simulate trend (inflation/demand trend factor)
    if 'year' in df.columns:
        df['price_trend'] = (df['year'] - df['year'].min()) * 1.5 + (df['price'] / 10)
    else:
        df['price_trend'] = np.random.uniform(0.9, 1.1, len(df))
    
    # 7. Feature: book_age
    current_year = 2026
    if 'year' in df.columns:
        df['book_age'] = current_year - df['year']
    else:
        df['book_age'] = 0
    
    # 8. Feature: seasonal_factor (Mock based on publication trends)
    df['seasonal_factor'] = np.random.uniform(0.95, 1.05, len(df))
    
    return df

def train_mega_ensemble():
    """
    Main training script implementing:
    - XGBoost, Random Forest, Linear Regression Voting Ensemble
    - GridSearchCV 5-Fold Cross Validation
    - Pipeline preprocessing
    """
    print("[INFO] Loading dataset...")
    df = pd.read_csv('data/dataset.csv')
    
    # Feature Engineering
    df = perform_feature_engineering(df)
    
    # Define Target and Features
    y = df['price']
    
    # Features mentioned in requirement for input: Store, Rating, Delivery, Demand
    # But for a robust model we use more
    X = df.drop(['price', 'title', 'price_per_rating', 'price_trend'], axis=1) # Drop target and leakies
    
    # Identify column types
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Preprocessing Pipeline
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, numerical_cols),
        ('cat', cat_transformer, categorical_cols)
    ])
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("[INFO] Hyperparameter Tuning via GridSearchCV (RF and XGB components)...")
    
    # We will tune components and then ensemble them
    # Component 1: Random Forest
    rf = RandomForestRegressor(random_state=42)
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    # Use a mini-pipeline for GridSearch to avoid double-fitting the preprocessor
    rf_pipe = Pipeline([('pre', preprocessor), ('rf', rf)])
    rf_gs = GridSearchCV(rf_pipe, {'rf__' + k: v for k, v in rf_params.items()}, cv=5, n_jobs=-1, scoring='r2')
    rf_gs.fit(X_train, y_train)
    best_rf = rf_gs.best_estimator_.named_steps['rf']
    # Fit preprocessor once for XGB grid search reuse
    X_train_proc = preprocessor.fit_transform(X_train)
    best_rf = rf_gs.best_estimator_
    
    # Component 2: XGBoost
    xgb = XGBRegressor(random_state=42)
    xgb_params = {
        'n_estimators': [100, 200],
        'max_depth': [6, 10],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0]
    }
    xgb_gs = GridSearchCV(xgb, xgb_params, cv=5, n_jobs=-1, scoring='r2')
    xgb_gs.fit(X_train_proc, y_train)
    best_xgb = xgb_gs.best_estimator_
    
    # Component 3: Linear Regression (Simple)
    lr = LinearRegression()
    
    print("[INFO] Creating VotingRegressor Ensemble...")
    # Voting Ensemble
    ensemble = VotingRegressor(estimators=[
        ('lr', lr),
        ('rf', best_rf),
        ('xgb', best_xgb)
    ])
    
    # Final Pipeline
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', ensemble)
    ])
    
    # Evaluate with Cross Validation
    cv_scores = cross_val_score(full_pipeline, X, y, cv=5, scoring='r2')
    print(f"[SUCCESS] 5-Fold Cross-Val Mean R2: {np.mean(cv_scores):.4f}")
    
    # Train full model
    start_time = time.time()
    full_pipeline.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"[COMPLETE] Model training finished in {training_time:.2f}s")
    
    # Evaluation
    y_pred = full_pipeline.predict(X_test)
    
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred)
    }
    
    print("\n[MODEL EVALUATION]")
    for m, val in metrics.items():
        print(f"  - {m}: {val:.4f}")
    
    # Model Visualizations
    # 1. Actual vs Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='#764ba2')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted Prices')
    plt.savefig('reports/visualizations/actual_vs_predicted_mega.png')
    plt.close()
    
    # 2. Residual Plot
    plt.figure(figsize=(10, 6))
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5, color='#667eea')
    plt.axhline(0, color='red', linestyle='--', lw=2)
    plt.xlabel('Predicted Price')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.savefig('reports/visualizations/residual_plot_mega.png')
    plt.close()
    
    # 3. Error Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, color='#f093fb')
    plt.title('Error (Residual) Distribution')
    plt.savefig('reports/visualizations/error_distribution_mega.png')
    plt.close()
    
    # 4. Feature Importance (Using RF Component as Proxy)
    # Get feature names after one-hot encoding
    cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols).tolist()
    feature_names = numerical_cols + cat_feature_names
    
    importances = best_rf.feature_importances_
    # Limit to top 15 for visibility
    indices = np.argsort(importances)[-15:]
    
    plt.figure(figsize=(10, 8))
    plt.title('Top 15 Feature Importances (Random Forest Proxy)')
    plt.barh(range(len(indices)), importances[indices], align='center', color='#3cba92')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.savefig('reports/visualizations/feature_importance_mega.png')
    plt.close()
    
    # Persistence
    joblib.dump(full_pipeline, 'models/price_model.pkl')
    print(f"\n[INFO] Model saved to models/price_model.pkl")
    
    # Save metrics for app to display
    with open('reports/model_performance_mega.txt', 'w') as f:
        f.write(f"Accuracy Metric Report\n")
        f.write(f"======================\n")
        f.write(f"MAE: {metrics['MAE']:.4f}\n")
        f.write(f"MSE: {metrics['MSE']:.4f}\n")
        f.write(f"RMSE: {metrics['RMSE']:.4f}\n")
        f.write(f"R2 Score: {metrics['R2']:.4f}\n")

if __name__ == "__main__":
    train_mega_ensemble()
