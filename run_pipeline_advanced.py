"""
Master Pipeline Runner

This script runs the complete end-to-end pipeline:
1. Data Cleaning
2. Feature Engineering
3. EDA (Exploratory Data Analysis)
4. Model Training
5. Model Evaluation

Author: Kunal Patel
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_cleaning import DataCleaner
from src.feature_engineering_advanced import AdvancedFeatureEngineer
from src.eda import EDAAnalyzer
from src.train_advanced import AdvancedTrainer


def run_complete_pipeline():
    """
    Run the complete end-to-end pipeline.
    """
    print("\n" + "="*70)
    print(" "*15 + "AMAZON BESTSELLING BOOKS")
    print(" "*10 + "COMPLETE ANALYTICS PIPELINE")
    print("="*70 + "\n")
    
    success_steps = []
    failed_steps = []
    
    # Step 1: Data Cleaning
    print("\n" + "="*70)
    print("STEP 1/5: DATA CLEANING")
    print("="*70)
    
    cleaner = DataCleaner("data/raw.csv", "data/cleaned.csv")
    if cleaner.run_pipeline():
        success_steps.append("Data Cleaning")
    else:
        failed_steps.append("Data Cleaning")
        print("\n[ERROR] Data cleaning failed. Stopping pipeline.")
        return False
    
    # Step 2: Advanced Feature Engineering
    print("\n" + "="*70)
    print("STEP 2/5: ADVANCED FEATURE ENGINEERING")
    print("="*70)
    
    # Engineer features from CLEANED data (to benefit from cleaning)
    engineer = AdvancedFeatureEngineer("data/cleaned.csv", "data/features.csv")
    if engineer.run():
        success_steps.append("Feature Engineering")
    else:
        failed_steps.append("Feature Engineering")
        print("\n[ERROR] Feature engineering failed. Stopping pipeline.")
        return False
    
    # Step 3: EDA
    print("\n" + "="*70)
    print("STEP 3/5: EXPLORATORY DATA ANALYSIS")
    print("="*70)
    
    analyzer = EDAAnalyzer("data/cleaned.csv", "reports")
    if analyzer.run_pipeline():
        success_steps.append("EDA")
    else:
        failed_steps.append("EDA")
        print("\n[WARNING] EDA failed but continuing pipeline...")
    
    # Step 4: Advanced Model Training
    print("\n" + "="*70)
    print("STEP 4/5: ADVANCED MODEL TRAINING")
    print("="*70)
    
    trainer = AdvancedTrainer("data/features.csv")
    try:
        trainer.run()
        success_steps.append("Model Training")
    except Exception as e:
        failed_steps.append("Model Training")
        print(f"\n[ERROR] Model training failed: {str(e)}")
        return False
    
    # Step 5: Model Evaluation (Covered in Training)
    print("\n" + "="*70)
    print("STEP 5/5: MODEL EVALUATION")
    print("="*70)
    print("[INFO] Evaluation metrics logged to MLflow and reports/model_performance.txt")
    success_steps.append("Model Evaluation")
    
    # Final Summary
    print("\n" + "="*70)
    print(" "*20 + "PIPELINE SUMMARY")
    print("="*70 + "\n")
    
    print("[OK] SUCCESSFUL STEPS:")
    for step in success_steps:
        print(f"  [OK] {step}")
    
    if failed_steps:
        print("\n[FAILED] FAILED STEPS:")
        for step in failed_steps:
            print(f"  [FAILED] {step}")
    
    print("\n" + "="*70)
    print(" "*15 + "PIPELINE EXECUTION COMPLETE")
    print("="*70 + "\n")
    
    print("OUTPUTS:")
    print(f"  - Cleaned Data: data/cleaned.csv")
    print(f"  - Features Data: data/features.csv")
    print(f"  - Models: models/price_model_advanced.pkl, models/success_model_advanced.pkl")
    print(f"  - Visualizations: reports/visualizations/")
    print(f"  - Insights: reports/insights.txt")
    print(f"  - Model Performance: reports/model_performance.txt")
    
    print("\n" + "="*70 + "\n")
    
    return len(failed_steps) == 0


if __name__ == "__main__":
    import time
    start_time = time.time()
    
    success = run_complete_pipeline()
    
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
    
    if success:
        print("\n[OK] ALL STEPS COMPLETED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print("\n[FAILED] PIPELINE COMPLETED WITH ERRORS")
        sys.exit(1)