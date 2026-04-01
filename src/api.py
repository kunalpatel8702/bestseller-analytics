"""
FastAPI Prediction Endpoint
Serves model predictions via REST API.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.predict_advanced import AdvancedPredictor
import uvicorn
import os

app = FastAPI(title="Book Price Strategy API", version="2.0")

# Initialize Predictor
predictor = AdvancedPredictor()
if not predictor.load_artifacts():
    print("WARNING: Model artifacts not found.")

class BookInput(BaseModel):
    title: str = "Example Book"
    genre: str = "Technology"
    author: str = "Jane Doe"
    rating: float = 4.5
    reviews_count: int = 1000
    year: int = 2024
    publisher: str = "Unknown"

@app.get("/")
def read_root():
    return {"status": "online", "model": "CatBoost Advanced"}

@app.post("/predict")
def predict_price(book: BookInput):
    if not predictor.price_model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert Pydantic to dict
        book_dict = book.dict()
        
        # Get raw market price
        market_price = predictor.predict_market_price(book_dict)
        
        # Get optimization (causal)
        opt_result = predictor.optimize_price(book_dict)
        
        return {
            "market_price": float(market_price),
            "optimal_price": float(opt_result['optimal_price']),
            "max_expected_value_score": float(opt_result['max_expected_value']),
            "success_probability": float(opt_result['success_probability']),
            "strategy": "Premium" if opt_result['optimal_price'] > market_price else "Penetration"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
