import pandas as pd
import numpy as np
import os
import joblib
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

# Create output directories
os.makedirs('reports/visualizations', exist_ok=True)
os.makedirs('models', exist_ok=True)

def perform_feature_engineering(df):
    """Basic feature engineering for robustness."""
    df = df.copy()
    if 'discount_percent' not in df.columns:
        df['discount_percent'] = 20.0
    if 'delivery_cost' not in df.columns:
        df['delivery_cost'] = 40.0
    
    # Simple proxies for missing feature columns in training data to match prediction requirements
    if 'reviews_count' in df.columns:
        df['demand_score'] = (df['reviews_count'] / 50000) * 100
        df['popularity_score'] = (df['rating'] * 15) + (df['reviews_count'] / 2500)
    else:
        df['demand_score'] = 50.0
        df['popularity_score'] = 60.0
        
    current_year = 2026
    if 'year' in df.columns:
        df['book_age'] = current_year - df['year']
    else:
        df['book_age'] = 1.0
        
    df['seasonal_factor'] = 1.0
    return df

def train_mega_ensemble():
    print("[INFO] Loading dataset...")
    df = pd.read_csv('data/raw.csv')
    df = perform_feature_engineering(df)
    
    y = df['price']
    # Select feature set strictly
    X = df[['genre', 'author', 'publisher', 'rating', 'reviews_count', 'year']]
    # Add dummy 'store' and 'delivery_type' if missing, just to train on them if they exist in schema
    for col in ['store', 'delivery_type']:
        if col not in X.columns:
            X[col] = 'Unknown'
            
    categorical_cols = X.select_dtypes(include=['object', 'string']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"[INFO] Categorical columns: {categorical_cols}")
    print(f"[INFO] Numerical columns: {numerical_cols}")
    
    # Robust Preprocessing
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
    
    # Models
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    lr = LinearRegression()
    
    ensemble = VotingRegressor(estimators=[
        ('lr', lr),
        ('rf', rf),
        ('xgb', xgb)
    ])
    
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', ensemble)
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("[INFO] Training VotingRegressor Ensemble...")
    full_pipeline.fit(X_train, y_train)
    
    y_pred = full_pipeline.predict(X_test)
    print(f"[SUCCESS] Test R2: {r2_score(y_test, y_pred):.4f}")
    print(f"[SUCCESS] Test MAE: ${mean_absolute_error(y_test, y_pred):.2f}")
    
    # Save precisely to the expected path
    joblib.dump(full_pipeline, 'models/price_model.pkl')
    print(f"[INFO] Model successfully saved to models/price_model.pkl")

if __name__ == "__main__":
    train_mega_ensemble()
