"""
Model Evaluation Module for Amazon Bestselling Books Price Prediction

This module:
- Loads trained model
- Evaluates on test set
- Generates detailed performance metrics
- Creates prediction vs actual plots
- Analyzes prediction errors

Author: Kunal Patel
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# Set plot style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)


class ModelEvaluator:
    """
    Model evaluation pipeline for price prediction.
    Generates comprehensive performance analysis.
    """
    
    def __init__(self, data_path: str, model_path: str, feature_names_path: str, output_dir: str):
        """
        Initialize ModelEvaluator.
        
        Args:
            data_path (str): Path to feature-engineered CSV file
            model_path (str): Path to trained model
            feature_names_path (str): Path to feature names pickle
            output_dir (str): Directory to save evaluation outputs
        """
        self.data_path = data_path
        self.model_path = model_path
        self.feature_names_path = feature_names_path
        self.output_dir = Path(output_dir)
        self.df = None
        self.model = None
        self.feature_names = None
        self.X = None
        self.y = None
        self.y_pred = None
        
    def load_model(self):
        """Load trained model from disk."""
        try:
            self.model = joblib.load(self.model_path)
            print(f"[INFO] Loaded model from {self.model_path}")
            return True
        except FileNotFoundError:
            print(f"[ERROR] Model file not found: {self.model_path}")
            print("[INFO] Please run train.py first")
            return False
        except Exception as e:
            print(f"[ERROR] Failed to load model: {str(e)}")
            return False
    
    def load_feature_names(self):
        """Load feature names from disk."""
        try:
            self.feature_names = joblib.load(self.feature_names_path)
            print(f"[INFO] Loaded {len(self.feature_names)} feature names")
            return True
        except FileNotFoundError:
            print(f"[ERROR] Feature names file not found: {self.feature_names_path}")
            return False
        except Exception as e:
            print(f"[ERROR] Failed to load feature names: {str(e)}")
            return False
    
    def load_data(self):
        """Load feature-engineered data from CSV file."""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"[INFO] Loaded dataset: {self.df.shape[0]} rows")
            return True
        except FileNotFoundError:
            print(f"[ERROR] File not found: {self.data_path}")
            return False
        except Exception as e:
            print(f"[ERROR] Failed to load data: {str(e)}")
            return False
    
    def prepare_data(self):
        """Prepare data for evaluation."""
        try:
            # Select features
            self.X = self.df[self.feature_names]
            self.y = self.df['price']
            print(f"[INFO] Prepared {len(self.X)} samples for evaluation")
            return True
        except KeyError as e:
            print(f"[ERROR] Missing feature column: {str(e)}")
            return False
    
    def make_predictions(self):
        """Generate predictions using trained model."""
        try:
            self.y_pred = self.model.predict(self.X)
            print(f"[INFO] Generated predictions for {len(self.y_pred)} samples")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to make predictions: {str(e)}")
            return False
    
    def calculate_metrics(self):
        """Calculate comprehensive performance metrics."""
        print("\n" + "="*60)
        print("PERFORMANCE METRICS")
        print("="*60 + "\n")
        
        # Basic metrics
        mae = mean_absolute_error(self.y, self.y_pred)
        rmse = np.sqrt(mean_squared_error(self.y, self.y_pred))
        r2 = r2_score(self.y, self.y_pred)
        
        # Additional metrics
        mape = np.mean(np.abs((self.y - self.y_pred) / self.y)) * 100
        median_ae = np.median(np.abs(self.y - self.y_pred))
        
        print(f"Mean Absolute Error (MAE):     ${mae:.2f}")
        print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
        print(f"R² Score:                       {r2:.4f}")
        print(f"Mean Absolute Percentage Error: {mape:.2f}%")
        print(f"Median Absolute Error:          ${median_ae:.2f}")
        
        # Prediction range
        print(f"\nPrediction Range:")
        print(f"  Minimum: ${self.y_pred.min():.2f}")
        print(f"  Maximum: ${self.y_pred.max():.2f}")
        print(f"  Mean:    ${self.y_pred.mean():.2f}")
        
        # Actual range
        print(f"\nActual Range:")
        print(f"  Minimum: ${self.y.min():.2f}")
        print(f"  Maximum: ${self.y.max():.2f}")
        print(f"  Mean:    ${self.y.mean():.2f}")
        
        print("\n" + "="*60 + "\n")
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'median_ae': median_ae
        }
    
    def analyze_errors(self):
        """Analyze prediction errors."""
        print("="*60)
        print("ERROR ANALYSIS")
        print("="*60 + "\n")
        
        # Calculate errors
        errors = self.y - self.y_pred
        abs_errors = np.abs(errors)
        pct_errors = (errors / self.y) * 100
        
        # Error statistics
        print("Error Statistics:")
        print(f"  Mean Error:     ${errors.mean():.2f}")
        print(f"  Std Error:      ${errors.std():.2f}")
        print(f"  Min Error:      ${errors.min():.2f}")
        print(f"  Max Error:      ${errors.max():.2f}")
        
        # Percentage of predictions within tolerance
        within_1_dollar = (abs_errors <= 1).sum() / len(abs_errors) * 100
        within_2_dollars = (abs_errors <= 2).sum() / len(abs_errors) * 100
        within_5_dollars = (abs_errors <= 5).sum() / len(abs_errors) * 100
        
        print(f"\nPrediction Accuracy:")
        print(f"  Within $1:  {within_1_dollar:.1f}%")
        print(f"  Within $2:  {within_2_dollars:.1f}%")
        print(f"  Within $5:  {within_5_dollars:.1f}%")
        
        # Identify worst predictions
        worst_indices = abs_errors.nlargest(5).index
        print(f"\nTop 5 Worst Predictions:")
        print("-" * 60)
        for idx in worst_indices:
            print(f"  Actual: ${self.y.iloc[idx]:.2f}, Predicted: ${self.y_pred[idx]:.2f}, Error: ${errors.iloc[idx]:.2f}")
        
        print("\n" + "="*60 + "\n")
    
    def create_visualizations(self):
        """Create evaluation visualizations."""
        print("[INFO] Creating visualizations...")
        
        # Create output directory
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Predicted vs Actual scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(self.y, self.y_pred, alpha=0.5, edgecolors='k', linewidths=0.5)
        plt.plot([self.y.min(), self.y.max()], [self.y.min(), self.y.max()], 'r--', lw=2, label='Perfect Prediction')
        plt.xlabel('Actual Price ($)', fontsize=12)
        plt.ylabel('Predicted Price ($)', fontsize=12)
        plt.title('Predicted vs Actual Prices', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(viz_dir / 'predicted_vs_actual.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Residual plot
        residuals = self.y - self.y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_pred, residuals, alpha=0.5, edgecolors='k', linewidths=0.5)
        plt.axhline(y=0, color='r', linestyle='--', lw=2)
        plt.xlabel('Predicted Price ($)', fontsize=12)
        plt.ylabel('Residuals ($)', fontsize=12)
        plt.title('Residual Plot', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(viz_dir / 'residual_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Error distribution
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero Error')
        plt.xlabel('Prediction Error ($)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(viz_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Absolute error by price range
        price_bins = pd.cut(self.y, bins=5)
        abs_errors = np.abs(residuals)
        error_by_price = pd.DataFrame({'price_range': price_bins, 'abs_error': abs_errors})
        
        plt.figure(figsize=(12, 6))
        error_by_price.boxplot(column='abs_error', by='price_range', ax=plt.gca())
        plt.xlabel('Price Range ($)', fontsize=12)
        plt.ylabel('Absolute Error ($)', fontsize=12)
        plt.title('Prediction Error by Price Range', fontsize=14, fontweight='bold')
        plt.suptitle('')  # Remove default title
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(viz_dir / 'error_by_price_range.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[SUCCESS] Visualizations saved to {viz_dir}")
    
    def save_predictions(self):
        """Save predictions to CSV file."""
        try:
            # Create predictions dataframe
            predictions_df = pd.DataFrame({
                'title': self.df['title'],
                'author': self.df['author'],
                'genre': self.df['genre'],
                'actual_price': self.y,
                'predicted_price': self.y_pred,
                'error': self.y - self.y_pred,
                'abs_error': np.abs(self.y - self.y_pred)
            })
            
            # Sort by absolute error (worst predictions first)
            predictions_df = predictions_df.sort_values('abs_error', ascending=False)
            
            # Save to CSV
            output_path = self.output_dir / 'predictions.csv'
            predictions_df.to_csv(output_path, index=False)
            
            print(f"[SUCCESS] Predictions saved to {output_path}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to save predictions: {str(e)}")
            return False
    
    def run_pipeline(self):
        """
        Execute complete evaluation pipeline.
        
        Returns:
            bool: True if successful, False otherwise
        """
        print("\n" + "="*60)
        print("STARTING MODEL EVALUATION PIPELINE")
        print("="*60 + "\n")
        
        # Load model
        if not self.load_model():
            return False
        
        # Load feature names
        if not self.load_feature_names():
            return False
        
        # Load data
        if not self.load_data():
            return False
        
        # Prepare data
        if not self.prepare_data():
            return False
        
        # Make predictions
        if not self.make_predictions():
            return False
        
        # Calculate metrics
        self.calculate_metrics()
        
        # Analyze errors
        self.analyze_errors()
        
        # Create visualizations
        self.create_visualizations()
        
        # Save predictions
        self.save_predictions()
        
        return True


def main():
    """Main execution function."""
    # Define paths
    data_path = "data/cleaned.csv"
    model_path = "models/price_model.pkl"
    feature_names_path = "models/feature_names.pkl"
    output_dir = "reports"
    
    # Initialize and run evaluator
    evaluator = ModelEvaluator(data_path, model_path, feature_names_path, output_dir)
    success = evaluator.run_pipeline()
    
    if success:
        print("\n[SUCCESS] Model evaluation completed successfully!")
    else:
        print("\n[FAILED] Model evaluation encountered errors.")
    
    return success


if __name__ == "__main__":
    main()
