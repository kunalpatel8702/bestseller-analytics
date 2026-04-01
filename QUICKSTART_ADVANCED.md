# Quick Start Guide

## Amazon Bestselling Books: EDA and Price Prediction

### 🚀 Getting Started in 5 Minutes

---

## Option 1: Run with Sample Data (Recommended for Testing)

If you don't have a dataset, generate sample data first:

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Generate sample dataset
python generate_sample_data.py

# Step 3: Run complete pipeline
python run_pipeline.py
```

**That's it!** The complete pipeline will run automatically.

---

## Option 2: Run with Your Own Data

### 1. Prepare Your Data

Place your CSV file at `data/raw.csv` with these columns:

**Required:**
- `title` - Book title
- `author` - Author name
- `genre` - Book genre/category
- `price` - Book price (can include $ symbol)
- `rating` - Customer rating (0-5)
- `reviews_count` - Number of reviews

**Optional:**
- `year` - Publication year
- `publisher` - Publisher name

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Pipeline

**Option A: Run Everything at Once**
```bash
python run_pipeline.py
```

**Option B: Run Step-by-Step**
```bash
# Step 1: Clean data
python src/data_cleaning.py

# Step 2: Engineer features
python src/feature_engineering.py

# Step 3: Run EDA
python src/eda.py

# Step 4: Train model
python src/train.py

# Step 5: Evaluate model
python src/evaluate.py
```

---

## 📊 Outputs

After running the pipeline, you'll find:

### Data
- `data/cleaned.csv` - Cleaned and feature-engineered dataset

### Models
- `models/price_model.pkl` - Trained price prediction model
- `models/feature_names.pkl` - Feature names for prediction

### Reports
- `reports/insights.txt` - Top 10 data insights
- `reports/model_performance.txt` - Model evaluation metrics
- `reports/predictions.csv` - Predictions on test set
- `reports/visualizations/` - All EDA plots (10+ charts)

---

## 🔮 Interactive Dashboard (New!)

To launch the beautiful, reactive web interface:

```bash
# Easy launcher
start_dashboard.bat

# Or manual command
python -m streamlit run app.py
```

## 🧠 Python Prediction API

Use the trained model to predict book prices in your own scripts:

```python
from src.predict import predict_price, get_pricing_recommendation

# Example book
book = {
    'genre': 'fiction',
    'rating': 4.5,
    'reviews_count': 5000,
    'author': 'Jane Doe',
    'year': 2023
}

# Get prediction
price = predict_price(book)
print(f"Predicted Price: ${price:.2f}")

# Get pricing recommendation
recommendation = get_pricing_recommendation(book)
print(f"Pricing Band: {recommendation['pricing_band']}")
print(f"Reasoning: {recommendation['reasoning']}")
```

---

## 📓 Interactive Analysis

For interactive exploration, use the Jupyter notebook:

```bash
jupyter notebook notebooks/eda.ipynb
```

---

## 🎯 What Each Module Does

| Module | Purpose |
|--------|---------|
| `data_cleaning.py` | Cleans raw data, handles missing values, removes duplicates |
| `feature_engineering.py` | Creates derived features for modeling |
| `eda.py` | Generates visualizations and insights |
| `train.py` | Trains and compares multiple models |
| `evaluate.py` | Evaluates model performance with detailed metrics |
| `predict.py` | Makes predictions and generates pricing recommendations |

---

## ⚠️ Troubleshooting

### "File not found: data/raw.csv"
**Solution:** Either generate sample data (`python generate_sample_data.py`) or place your CSV at `data/raw.csv`

### "Module not found"
**Solution:** Install dependencies: `pip install -r requirements.txt`

### "Small dataset warning"
**Solution:** This is normal for datasets < 100 samples. Model will still work but performance may be limited.

---

## 📈 Expected Results

With the sample dataset (500 books):
- **Model R²:** ~0.70-0.85
- **MAE:** ~$2-4
- **Processing Time:** 30-60 seconds

---

## 🎓 For Hiring Managers

This project demonstrates:
- ✅ Production-ready code structure
- ✅ Comprehensive data cleaning and validation
- ✅ Feature engineering best practices
- ✅ Model comparison and selection
- ✅ Automated insights generation
- ✅ Business-focused recommendations
- ✅ Complete documentation
- ✅ Error handling and warnings
- ✅ Modular, reusable code

---

## 📞 Next Steps

1. ✅ Run the pipeline with sample data
2. ✅ Review the generated insights and visualizations
3. ✅ Examine the model performance report
4. ✅ Try making predictions with the trained model
5. ✅ Explore the Jupyter notebook for interactive analysis

---

**Ready to start?**

```bash
pip install -r requirements.txt
python generate_sample_data.py
python run_pipeline.py
```

**Questions?** Check the main README.md for detailed documentation.
