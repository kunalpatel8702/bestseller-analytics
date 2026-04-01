import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime

class BookPriceEnsemblePredictor:
    def __init__(self, model_path='models/price_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Automatically load model when app starts."""
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                print(f"[SUCCESS] Ensemble Model loaded from {self.model_path}")
                return True
            except Exception as e:
                print(f"[ERROR] Model loading failure: {str(e)}")
        return False

    def _apply_fe(self, df):
        """Same feature engineering as training script."""
        current_year = 2026
        
        # 1. discount_percent (Average fallback)
        if 'discount_percent' not in df.columns:
            df['discount_percent'] = 20.0 
            
        # 2. delivery_cost (Average fallback)
        if 'delivery_cost' not in df.columns:
            df['delivery_cost'] = 40.0
            
        # 3. demand_score (Calculated or input)
        # Using a default scaling based on known max reviews
        if 'reviews_count' in df.columns:
            df['demand_score'] = (df['reviews_count'] / 50000) * 100
        else:
            df['demand_score'] = 50.0
            
        # 4. popularity_score
        if 'rating' in df.columns and 'reviews_count' in df.columns:
            df['popularity_score'] = (df['rating'] * 15) + (df['reviews_count'] / 2500)
        else:
            df['popularity_score'] = 60.0
            
        # 5. book_age
        if 'year' in df.columns:
            df['book_age'] = current_year - df['year']
        else:
            df['book_age'] = 1.0
            
        # 6. seasonal_factor (Assume neutral)
        df['seasonal_factor'] = 1.0
        
        return df

    def predict_mega(self, input_data):
        """
        When user inputs features: Predict price instantly and Display confidence score.
        Accepts dict or dataframe.
        """
        if self.model is None:
            return {"success": False, "error": "Model loading failure"}
            
        try:
            # Prepare input
            if isinstance(input_data, dict):
                df_input = pd.DataFrame([input_data])
            else:
                df_input = input_data.copy()
            
            # Feature check and default synthesis
            required_cols = ['year', 'reviews_count', 'rating', 'genre', 'publisher', 'author', 'store']
            for col in required_cols:
                if col not in df_input.columns:
                    # Provide neutral defaults to prevent crash
                    if col in ['rating', 'reviews_count', 'year']:
                        df_input[col] = 0
                    else:
                        df_input[col] = 'Unknown'
            
            # Apply FE
            df_input = self._apply_fe(df_input)
            
            # PREDICT
            prediction = self.model.predict(df_input)[0]
            
            # Calculate Confidence Score
            # By getting predictions from each member of the ensemble
            # pipeline.steps[-1][1] is the VotingRegressor
            ensemble_model = self.model.named_steps['model']
            preprocessor = self.model.named_steps['preprocessor']
            
            proc_data = preprocessor.transform(df_input)
            
            individual_preds = []
            for name, estimator in ensemble_model.named_estimators_.items():
                individual_preds.append(estimator.predict(proc_data)[0])
                
            # Confidence Score = 1 - (Relative Stan Dev)
            # Higher agreement = higher confidence
            mean_pred = np.mean(individual_preds)
            std_pred = np.std(individual_preds)
            
            # Scale score between 0 and 100
            # A 20% deviation relative to price drops score significantly
            cv = std_pred / (abs(mean_pred) + 1e-6)
            confidence_score = max(0, min(100, (1 - (cv * 2)) * 100))
            
            return {
                "success": True,
                "predicted_price": round(prediction, 2),
                "confidence_score": round(confidence_score, 1),
                "model_components": {
                    "LR": round(individual_preds[0], 2),
                    "RF": round(individual_preds[1], 2),
                    "XGB": round(individual_preds[2], 2)
                }
            }
            
        except Exception as e:
            return {"success": False, "error": f"Prediction failure: {str(e)}"}

if __name__ == "__main__":
    # Test
    predictor = BookPriceEnsemblePredictor()
    test_input = {
        'genre': 'Business', 'author': 'James Clear', 'publisher': 'Penguin',
        'rating': 4.7, 'reviews_count': 15000, 'year': 2020, 'store': 'Amazon'
    }
    print(predictor.predict_mega(test_input))
