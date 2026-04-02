"""
Model Training Module for Amazon Bestselling Books Price Prediction

This module:
- Trains multiple regression models (Ridge, Random Forest)
- Compares model performance using MAE, RMSE, R²
- Automatically selects best model
- Saves trained model for production use
- Generates feature importance analysis

Author: Kunal Patel
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Model training pipeline for book price prediction.
    Trains multiple models and selects the best performer.
    """
    
    def __init__(self, data_path: str, model_output_path: str, report_output_path: str):
        """
        Initialize ModelTrainer.
        
        Args:
            data_path (str): Path to feature-engineered CSV file
            model_output_path (str): Path to save trained model
            report_output_path (str): Path to save performance report
        """
        self.data_path = data_path
        self.model_output_path = model_output_path
        self.report_output_path = report_output_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        
    def load_data(self):
        """Load feature-engineered data from CSV file."""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"[INFO] Loaded dataset: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            return True
        except FileNotFoundError:
            print(f"[ERROR] File not found: {self.data_path}")
            print("[INFO] Please run feature_engineering.py first")
            return False
        except Exception as e:
            print(f"[ERROR] Failed to load data: {str(e)}")
            return False
    
    def prepare_features(self):
        """
        Prepare features for modeling.
        Select relevant features and create train/test split.
        """
        print("[INFO] Preparing features for modeling...")
        
        # Define feature columns (exclude target and non-predictive columns)
        exclude_cols = [
            'price',  # Target variable
            'title',  # Non-predictive
            'author',  # Use encoded version
            'genre',  # Use encoded version
            'publisher',  # Use encoded version (if exists)
            'price_category',  # Derived from target
            'rating_category',  # Categorical version
            'reviews_category',  # Categorical version
        ]
        
        # Select feature columns
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        
        # Ensure we have numeric features only
        X = self.df[feature_cols].select_dtypes(include=[np.number])
        y = self.df['price']
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        print(f"[INFO] Selected {len(self.feature_names)} features for modeling")
        print(f"[INFO] Features: {', '.join(self.feature_names[:10])}{'...' if len(self.feature_names) > 10 else ''}")
        
        # Train/test split (80/20)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"[INFO] Train set: {len(self.X_train)} samples")
        print(f"[INFO] Test set: {len(self.X_test)} samples")
        
        return True
    
    def train_ridge_regression(self):
        """
        Train Ridge Regression model.
        Ridge adds L2 regularization to prevent overfitting.
        """
        print("\n[INFO] Training Ridge Regression...")
        
        # Initialize model
        model = Ridge(alpha=1.0, random_state=42)
        
        # Train model
        model.fit(self.X_train, self.y_train)
        
        # Store model
        self.models['Ridge Regression'] = model
        
        print("[SUCCESS] Ridge Regression trained")
    
    def train_random_forest(self):
        """
        Train Random Forest Regressor.
        Ensemble of decision trees, robust to outliers.
        """
        print("\n[INFO] Training Random Forest...")
        
        # Initialize model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Train model
        model.fit(self.X_train, self.y_train)
        
        # Store model
        self.models['Random Forest'] = model
        
        print("[SUCCESS] Random Forest trained")
    
    def evaluate_model(self, model_name: str, model):
        """
        Evaluate model performance on train and test sets.
        
        Args:
            model_name (str): Name of the model
            model: Trained model object
        
        Returns:
            dict: Performance metrics
        """
        # Predictions
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)
        
        # Calculate metrics
        metrics = {
            'train_mae': mean_absolute_error(self.y_train, y_train_pred),
            'test_mae': mean_absolute_error(self.y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
            'train_r2': r2_score(self.y_train, y_train_pred),
            'test_r2': r2_score(self.y_test, y_test_pred),
        }
        
        # Cross-validation score (5-fold)
        cv_scores = cross_val_score(
            model, self.X_train, self.y_train,
            cv=5, scoring='r2', n_jobs=-1
        )
        metrics['cv_r2_mean'] = cv_scores.mean()
        metrics['cv_r2_std'] = cv_scores.std()
        
        return metrics
    
    def evaluate_all_models(self):
        """Evaluate all trained models and store results."""
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60 + "\n")
        
        for model_name, model in self.models.items():
            print(f"Evaluating {model_name}...")
            metrics = self.evaluate_model(model_name, model)
            self.results[model_name] = metrics
            
            # Print results
            print(f"\n{model_name} Performance:")
            print(f"  Train MAE:  ${metrics['train_mae']:.2f}")
            print(f"  Test MAE:   ${metrics['test_mae']:.2f}")
            print(f"  Train RMSE: ${metrics['train_rmse']:.2f}")
            print(f"  Test RMSE:  ${metrics['test_rmse']:.2f}")
            print(f"  Train R²:   {metrics['train_r2']:.4f}")
            print(f"  Test R²:    {metrics['test_r2']:.4f}")
            print(f"  CV R² (5-fold): {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")
            print()
    
    def select_best_model(self):
        """
        Select best model based on test R² score.
        Higher R² indicates better performance.
        """
        print("="*60)
        print("MODEL SELECTION")
        print("="*60 + "\n")
        
        # Find model with highest test R²
        best_score = -float('inf')
        best_name = None
        
        for model_name, metrics in self.results.items():
            if metrics['test_r2'] > best_score:
                best_score = metrics['test_r2']
                best_name = model_name
        
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        
        print(f"[OK] Best Model: {best_name}")
        print(f"[OK] Test R²: {best_score:.4f}")
        print(f"[OK] Test MAE: ${self.results[best_name]['test_mae']:.2f}")
        print(f"[OK] Test RMSE: ${self.results[best_name]['test_rmse']:.2f}")
        print()
    
    def analyze_feature_importance(self):
        """
        Analyze feature importance for tree-based models.
        For linear models, use coefficient magnitude.
        """
        print("="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60 + "\n")
        
        if 'Random Forest' in self.best_model_name:
            # Tree-based model: use feature_importances_
            importances = self.best_model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print("Top 10 Most Important Features:")
            print("-" * 60)
            for idx, row in feature_importance.head(10).iterrows():
                print(f"  {row['feature']:30s} {row['importance']:.4f}")
            
        elif 'Ridge' in self.best_model_name:
            # Linear model: use coefficient magnitude
            coefficients = np.abs(self.best_model.coef_)
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'coefficient': coefficients
            }).sort_values('coefficient', ascending=False)
            
            print("Top 10 Features by Coefficient Magnitude:")
            print("-" * 60)
            for idx, row in feature_importance.head(10).iterrows():
                print(f"  {row['feature']:30s} {row['coefficient']:.4f}")
        
        print("\n" + "="*60 + "\n")
        
        return feature_importance
    
    def save_model(self):
        """Save best model to disk using joblib."""
        try:
            # Create output directory if it doesn't exist
            output_dir = Path(self.model_output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            joblib.dump(self.best_model, self.model_output_path)
            print(f"[SUCCESS] Model saved to {self.model_output_path}")
            
            # Also save feature names for prediction
            feature_names_path = output_dir / "feature_names.pkl"
            joblib.dump(self.feature_names, feature_names_path)
            print(f"[SUCCESS] Feature names saved to {feature_names_path}")
            
            return True
        except Exception as e:
            print(f"[ERROR] Failed to save model: {str(e)}")
            return False
    
    def save_report(self):
        """Save performance report to text file."""
        try:
            # Create output directory if it doesn't exist
            output_dir = Path(self.report_output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(self.report_output_path, 'w') as f:
                f.write("="*60 + "\n")
                f.write("MODEL TRAINING REPORT\n")
                f.write("="*60 + "\n\n")
                
                f.write(f"Dataset: {len(self.df)} samples\n")
                f.write(f"Features: {len(self.feature_names)}\n")
                f.write(f"Train/Test Split: {len(self.X_train)}/{len(self.X_test)}\n\n")
                
                f.write("="*60 + "\n")
                f.write("MODEL COMPARISON\n")
                f.write("="*60 + "\n\n")
                
                for model_name, metrics in self.results.items():
                    f.write(f"{model_name}:\n")
                    f.write(f"  Test MAE:  ${metrics['test_mae']:.2f}\n")
                    f.write(f"  Test RMSE: ${metrics['test_rmse']:.2f}\n")
                    f.write(f"  Test R²:   {metrics['test_r2']:.4f}\n")
                    f.write(f"  CV R²:     {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}\n\n")
                
                f.write("="*60 + "\n")
                f.write("BEST MODEL\n")
                f.write("="*60 + "\n\n")
                f.write(f"Selected Model: {self.best_model_name}\n")
                f.write(f"Test R²: {self.results[self.best_model_name]['test_r2']:.4f}\n")
                f.write(f"Test MAE: ${self.results[self.best_model_name]['test_mae']:.2f}\n")
                f.write(f"Test RMSE: ${self.results[self.best_model_name]['test_rmse']:.2f}\n\n")
            
            print(f"[SUCCESS] Report saved to {self.report_output_path}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to save report: {str(e)}")
            return False
    
    def run_pipeline(self):
        """
        Execute complete model training pipeline.
        
        Returns:
            bool: True if successful, False otherwise
        """
        print("\n" + "="*60)
        print("STARTING MODEL TRAINING PIPELINE")
        print("="*60 + "\n")
        
        # Load data
        if not self.load_data():
            return False
        
        # Prepare features
        if not self.prepare_features():
            return False
        
        # Train models
        self.train_ridge_regression()
        self.train_random_forest()
        
        # Evaluate models
        self.evaluate_all_models()
        
        # Select best model
        self.select_best_model()
        
        # Analyze feature importance
        self.analyze_feature_importance()
        
        # Save model
        if not self.save_model():
            return False
        
        # Save report
        if not self.save_report():
            return False
        
        return True


def main():
    """Main execution function."""
    # Define paths
    data_path = "data/cleaned.csv"
    model_output_path = "models/price_model.pkl"
    report_output_path = "reports/model_performance.txt"
    
    # Initialize and run trainer
    trainer = ModelTrainer(data_path, model_output_path, report_output_path)
    success = trainer.run_pipeline()
    
    if success:
        print("\n[SUCCESS] Model training completed successfully!")
    else:
        print("\n[FAILED] Model training encountered errors.")
    
    return success


if __name__ == "__main__":
    main()
