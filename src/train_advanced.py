"""
Advanced Model Training Module (v2)
- Price Model: Predicts market price from book attributes.
- Success Model: Predicts bestseller status from book attributes AND PRICE (for causal inference).
- Logs experiments to MLflow.
"""
import pandas as pd
import numpy as np
import shap
import catboost as cb
import mlflow
import mlflow.catboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, recall_score, roc_auc_score
import joblib

class AdvancedTrainer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        mlflow.set_experiment("Book_Price_Success_V2")
        
    def prepare_data(self):
        """Prepare separate datasets for Price and Success models."""
        # Identify categorical features and fill NaNs
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                self.df[col] = self.df[col].fillna("Unknown")

        # 1. Price Model Data (Target: Price)
        # Features: Everything EXCEPT Price, Sales Rank, Is Success
        self.X_price = self.df.drop(['price', 'sales_rank', 'is_success'], axis=1)
        self.y_price = self.df['price']
        
        # 2. Success Model Data (Target: Is Success)
        # Features: Everything INCLUDING Price, EXCEPT Sales Rank, Is Success
        self.X_success = self.df.drop(['sales_rank', 'is_success'], axis=1)
        self.y_success = self.df['is_success']
        
        # Identify categorical features (same for both mostly, but X_success has Price which is numeric)
        self.cat_features_price = [col for col in self.X_price.columns if self.X_price[col].dtype == 'object']
        self.cat_features_success = [col for col in self.X_success.columns if self.X_success[col].dtype == 'object']
        
        # Split Data
        self.X_price_train, self.X_price_test, self.y_price_train, self.y_price_test = \
            train_test_split(self.X_price, self.y_price, test_size=0.2, random_state=42)
            
        self.X_success_train, self.X_success_test, self.y_success_train, self.y_success_test = \
            train_test_split(self.X_success, self.y_success, test_size=0.2, random_state=42)
            
    def train_price_model(self):
        """Train Price Regressor."""
        with mlflow.start_run(run_name="Price_Model_CatBoost"):
            print("[INFO] Training Price Model...")
            model = cb.CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, verbose=0)
            model.fit(self.X_price_train, self.y_price_train, cat_features=self.cat_features_price)
            
            # Evaluate
            preds = model.predict(self.X_price_test)
            mae = mean_absolute_error(self.y_price_test, preds)
            mlflow.log_metric("mae", mae)
            print(f"[SUCCESS] Price Model MAE: ${mae:.2f}")
            
            # Save
            joblib.dump(model, "models/price_model_advanced.pkl")
            mlflow.catboost.log_model(model, "price_model")
            return model
        
    def train_success_model(self):
        """Train Success Classifier (Simulating Effect of Price)."""
        with mlflow.start_run(run_name="Success_Model_CatBoost"):
            print("[INFO] Training Success Model...")
            model = cb.CatBoostClassifier(iterations=500, learning_rate=0.05, verbose=0)
            model.fit(self.X_success_train, self.y_success_train, cat_features=self.cat_features_success)
            
            # Evaluate
            preds = model.predict(self.X_success_test)
            probs = model.predict_proba(self.X_success_test)[:, 1]
            acc = accuracy_score(self.y_success_test, preds)
            auc = roc_auc_score(self.y_success_test, probs)
            
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("auc", auc)
            print(f"[SUCCESS] Success Model Accuracy: {acc:.2f}, AUC: {auc:.2f}")
            
            # Save
            joblib.dump(model, "models/success_model_advanced.pkl")
            mlflow.catboost.log_model(model, "success_model")
            return model
            
    def explain_model(self, model, X_sample, name="model"):
        """Save SHAP explainer."""
        print(f"[INFO] Explaining {name}...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_sample)
        joblib.dump(shap_values, f"models/shap_values_{name}.pkl")
        
    def run(self):
        self.prepare_data()
        price_model = self.train_price_model()
        success_model = self.train_success_model()
        
        # Explain
        self.explain_model(price_model, self.X_price_test, "price")
        # self.explain_model(success_model, self.X_success_test, "success") # Skip to save time if needed
        
        print("[COMPLETE] Advanced Training & Logging Finished.")

if __name__ == "__main__":
    trainer = AdvancedTrainer("data/features.csv")
    trainer.run()
