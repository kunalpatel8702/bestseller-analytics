# Amazon Bestselling Books: EDA and Price Prediction

## 📊 Project Overview

This project performs comprehensive exploratory data analysis (EDA) on Amazon bestselling books and builds a machine learning model to predict book prices based on various attributes such as genre, author, ratings, and review counts.

**Business Value**: Enable data-driven pricing strategies for publishers and retailers by understanding price drivers and generating automated pricing recommendations.

---

## 🎯 Problem Statement

Book pricing is complex and influenced by multiple factors including:
- Author reputation (bestseller count)
- Genre/category
- Customer ratings and review volume
- Market trends

This project aims to:
1. Identify key patterns in bestselling books
2. Build a predictive model for book pricing
3. Generate pricing recommendations based on market positioning

---

## 🏗️ Project Structure

```
project/
│
├── data/
│   ├── raw.csv              # Original dataset
│   └── cleaned.csv          # Processed dataset
│
├── notebooks/
│   └── eda.ipynb            # Exploratory Data Analysis
│
├── src/
│   ├── data_cleaning.py     # Data preprocessing module
│   ├── feature_engineering.py # Feature creation module
│   ├── train.py             # Model training pipeline
│   ├── evaluate.py          # Model evaluation utilities
│   └── predict.py           # Prediction and recommendation engine
│
├── models/
│   └── price_model.pkl      # Trained model artifact
│
├── reports/
│   ├── insights.txt         # Top 10 insights
│   └── visualizations/      # EDA plots
│
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

Place your CSV file in `data/raw.csv` with the following columns:
- **Required**: title, author, genre, price, rating, reviews_count
- **Optional**: year, publisher

### 3. Run Complete Pipeline

```bash
# Step 1: Clean data
python src/data_cleaning.py

# Step 2: Engineer features
python src/feature_engineering.py

# Step 3: Train model
python src/train.py

# Step 4: Evaluate model
python src/evaluate.py
```

### 4. Run Interactive Dashboard (New!)

Calculated price predictions and visualize market trends in a beautiful web interface.

```bash
# Run the Streamlit app
python -m streamlit run app.py

# Or use the helper script (Windows)
start_dashboard.bat
```

### 5. Make Predictions (Python Script)

```python
from src.predict import predict_price, get_pricing_recommendation

# Example prediction
book_features = {
    'genre': 'Fiction',
    'rating': 4.5,
    'reviews_count': 5000,
    'author': 'John Doe'
}

predicted_price = predict_price(book_features)
recommendation = get_pricing_recommendation(book_features, predicted_price)

print(f"Predicted Price: ${predicted_price:.2f}")
print(f"Recommendation: {recommendation}")
```

---

## 📈 Methodology

### Data Cleaning
- Remove currency symbols and standardize price format
- Convert reviews to integer, handle missing values
- Standardize genre text (lowercase, strip whitespace)
- Remove duplicate entries
- Robust handling of optional columns

### Feature Engineering
- **log_reviews**: Log transformation of review counts (handles skewness)
- **author_bestseller_count**: Number of bestsellers per author
- **year_features**: Optional time-based features if year is available
- **Categorical encoding**: Target encoding for high-cardinality features

### Model Training
Two models are trained and compared:
1. **Ridge Regression**: Linear model with L2 regularization
2. **Random Forest**: Ensemble tree-based model

**Evaluation Metrics**:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

The best-performing model is automatically selected and saved.

### Pricing Recommendations
Rule-based logic:
- **Premium Band**: Predicted price > genre average (top 25%)
- **Competitive Band**: Predicted price < genre average (bottom 25%)
- **Market Rate**: Within ±25% of genre average

---

## 📊 Key Insights

The analysis automatically generates top 10 insights including:
- Price distribution patterns
- Genre-wise pricing trends
- Impact of ratings and reviews on price
- Author influence on pricing
- Correlation between features

See `reports/insights.txt` for detailed findings.

---

## 🎯 Model Performance

Performance metrics are automatically logged during training:
- Training and test set performance
- Feature importance rankings
- Model comparison results

Check `reports/model_performance.txt` for detailed metrics.

---

## ⚠️ Limitations & Assumptions

1. **Data Quality**: Model performance depends on dataset size and quality
2. **Feature Availability**: Optional features (year, publisher) improve predictions
3. **Market Dynamics**: Model doesn't account for real-time market changes
4. **Genre Specificity**: Performance may vary across different genres
5. **Temporal Validity**: Pricing trends may change over time

**Automatic Warnings**:
- Small dataset warning (< 100 samples)
- Skewed distribution alerts
- Missing feature notifications

---

## 🔧 Technical Stack

- **Python 3.8+**
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn
- **Model Persistence**: joblib

---

## 📝 Future Enhancements

1. Deep learning models (neural networks)
2. Time series analysis for price trends
3. Sentiment analysis on book descriptions
4. A/B testing framework for pricing strategies
5. API deployment for real-time predictions

---

## 👥 Author

Kunal Patel

**Project Type**: Production-Ready Analytics Pipeline

**Last Updated**: February 2026

---

## 📄 License

This project is designed for portfolio and educational purposes.

---

## 🤝 Contributing

This is a portfolio project demonstrating end-to-end ML pipeline development. Feedback and suggestions are welcome!
