# Quick Start Guide

Get started with the Advanced Prediction System in 5 minutes!

## Step 1: Install Dependencies

```bash
cd /Users/noobmaster69/CascadeProjects/prediction-system
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install pmdarima
```

## Step 2: Run Quick Example

Test the system with sample data:

```bash
python example_usage.py quick
```

Expected output:
```
Quick Prediction Example
Input data: [10 12 15 14 18 20 22 25 23 28 30 32 35 33 38 40 42 45 43 48]
Next 3 predictions:
  Step 1: 50.23
  Step 2: 52.45
  Step 3: 54.67
```

## Step 3: Configure Your Website

Edit `config.yaml`:

```yaml
website:
  url: "https://your-website.com/data-page"
  selector: ".number-element"  # CSS selector for your data
```

Or create a `.env` file:

```bash
cp .env.example .env
# Edit .env with your website URL
```

## Step 4: Run Full Prediction

### With Sample Data
```bash
python main.py --steps 5 --visualize
```

### With Your Website Data
```bash
python main.py --scrape --steps 5 --visualize --save-models
```

### With Custom CSV File
```bash
python main.py --data-file your_data.csv --steps 10 --evaluate
```

## Step 5: View Results

Results will be displayed in the console and saved to:
- `outputs/predictions.png` - Static visualization
- `outputs/predictions_interactive.html` - Interactive chart
- `outputs/model_comparison.png` - Model comparison

## Common Use Cases

### 1. Predict Stock Prices
```python
import pandas as pd
from src.prediction_engine import PredictionEngine

# Load your stock data
df = pd.read_csv('stock_prices.csv')
prices = df['close'].values

# Predict
engine = PredictionEngine()
engine.initialize_models()
engine.train_models(prices)
result = engine.predict(prices, steps=5)

print("Next 5 day predictions:", result['prediction'])
```

### 2. Predict Website Metrics
```bash
# Configure your analytics URL in config.yaml
python main.py --scrape --steps 7 --visualize
```

### 3. Predict Sales Numbers
```python
from src.prediction_engine import PredictionEngine

sales_data = [100, 120, 115, 140, 135, 160, 155, 180]
engine = PredictionEngine()
engine.initialize_models()
engine.train_models(sales_data)
result = engine.predict(sales_data, steps=3)
```

## Understanding the Output

```
Ensemble Prediction: [50.23]
  - Combined prediction from all models

Confidence Interval (95%): [48.15, 52.31]
  - 95% confidence the true value will be in this range

Individual Model Predictions:
  lstm           : 50.45
  xgboost        : 50.12
  prophet        : 50.18
  arima          : 50.25

Model Weights:
  lstm           : 0.35  (35% contribution)
  xgboost        : 0.30  (30% contribution)
  prophet        : 0.20  (20% contribution)
  arima          : 0.15  (15% contribution)
```

## Troubleshooting

### "Not enough data to create sequences"
- You need at least 60 data points
- Solution: Reduce `sequence_length` in config.yaml to 20

### "Web scraping failed"
- Check if the URL is accessible
- Verify CSS selector is correct
- Try: `curl https://your-website.com` to test

### "Module not found"
- Make sure virtual environment is activated
- Run: `pip install -r requirements.txt`

## Next Steps

1. ✅ Run quick example
2. ✅ Configure your data source
3. ✅ Make your first prediction
4. 📊 Explore visualizations
5. 🎯 Fine-tune model parameters
6. 🚀 Deploy to production

## Need Help?

- Check `README.md` for detailed documentation
- Review `example_usage.py` for code examples
- Examine `config.yaml` for all configuration options

---

**You're ready to predict! 🎯**
