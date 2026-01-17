# Advanced Prediction System

A sophisticated Python-based prediction system that uses multiple advanced mathematical and machine learning models to predict future outcomes. The system extracts historical data from websites and applies ensemble learning techniques for optimal accuracy.

## Features

### 🎯 Multiple Prediction Models
- **LSTM (Long Short-Term Memory)**: Deep learning model for sequential data
- **Prophet**: Facebook's time series forecasting model
- **ARIMA**: Auto-regressive integrated moving average
- **XGBoost**: Gradient boosting for structured data
- **LightGBM**: Fast gradient boosting framework
- **Statistical Models**: Moving averages, exponential smoothing

### 🔄 Ensemble Learning
- Weighted averaging of multiple models
- Automatic weight optimization based on performance
- Confidence intervals for predictions
- Multiple ensemble strategies (weighted, median, trimmed mean)

### 🌐 Web Data Extraction
- Automated web scraping from your website
- Support for both HTML parsing and API endpoints
- Configurable selectors and retry mechanisms
- Historical data management

### 📊 Advanced Analytics
- Interactive visualizations with Plotly
- Static plots with Matplotlib/Seaborn
- Model comparison and performance metrics
- Error analysis and diagnostics
- Time series decomposition

## Installation

### 1. Clone or navigate to the project directory
```bash
cd /Users/noobmaster69/CascadeProjects/prediction-system
```

### 2. Create a virtual environment (recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install pmdarima for auto-ARIMA
```bash
pip install pmdarima
```

## Configuration

### 1. Copy the example environment file
```bash
cp .env.example .env
```

### 2. Edit `.env` with your website details
```
WEBSITE_URL=https://your-website.com/data
DATA_SELECTOR=.data-element
```

### 3. Customize `config.yaml`
- Set your website URL and CSS selector
- Adjust model parameters
- Configure prediction settings

## Usage

### Quick Start

#### Option 1: Using the main script
```bash
python main.py --steps 5 --visualize
```

#### Option 2: Using example scripts
```bash
# Quick prediction with sample data
python example_usage.py quick

# Full example with visualizations
python example_usage.py sample

# Web scraping example
python example_usage.py scrape
```

### Command Line Options

```bash
python main.py [OPTIONS]

Options:
  --config PATH       Path to configuration file (default: config.yaml)
  --scrape           Scrape data from website before prediction
  --steps N          Number of steps to predict ahead (default: 1)
  --visualize        Generate visualizations
  --save-models      Save trained models
  --load-models      Load previously trained models
  --evaluate         Evaluate model performance
  --data-file PATH   Use custom CSV file with 'value' column
```

### Examples

#### 1. Predict next 3 values with visualization
```bash
python main.py --steps 3 --visualize
```

#### 2. Scrape website data and predict
```bash
python main.py --scrape --steps 5 --visualize --save-models
```

#### 3. Use custom data file
```bash
python main.py --data-file my_data.csv --steps 10 --evaluate
```

#### 4. Load saved models and predict
```bash
python main.py --load-models --steps 1
```

## Python API Usage

```python
import numpy as np
from src.prediction_engine import PredictionEngine
from src.visualizer import PredictionVisualizer

# Create sample data
data = np.array([10, 12, 15, 14, 18, 20, 22, 25, 23, 28])

# Initialize prediction engine
engine = PredictionEngine(config_path='config.yaml')

# Initialize and train models
engine.initialize_models()
engine.train_models(data)

# Make predictions
result = engine.predict(data, steps=3)

# Access results
print("Ensemble Prediction:", result['prediction'])
print("Confidence Interval:", result['confidence_interval'])
print("Individual Models:", result['individual_predictions'])
print("Model Weights:", result['weights'])

# Visualize
visualizer = PredictionVisualizer()
visualizer.plot_predictions(data, result, result['confidence_interval'])
```

## Web Scraping Setup

### HTML Scraping
Update `config.yaml`:
```yaml
website:
  url: "https://your-website.com/data"
  selector: ".data-element"  # CSS selector for data elements
  data_type: "number"
```

### API Endpoint
```python
from src.data_scraper import APIDataExtractor

api = APIDataExtractor(
    api_url="https://your-api.com/data",
    api_key="your-api-key"
)
df = api.fetch_data()
```

## Model Configuration

### LSTM Parameters
```yaml
lstm:
  units: [128, 64, 32]  # Layer sizes
  dropout: 0.2
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
```

### XGBoost Parameters
```yaml
xgboost:
  n_estimators: 1000
  max_depth: 7
  learning_rate: 0.01
```

### Ensemble Settings
```yaml
ensemble:
  weights: "auto"  # or specify manual weights
  method: "weighted_average"  # or "median", "trimmed_mean"
```

## Output

The system generates:
- **Console Output**: Detailed predictions and statistics
- **Visualizations** (if --visualize flag is used):
  - `outputs/predictions.png`: Static prediction plot
  - `outputs/predictions_interactive.html`: Interactive plot
  - `outputs/model_comparison.png`: Model comparison chart
- **Saved Models** (if --save-models flag is used):
  - `models/saved/`: Trained model files

## Performance Metrics

The system evaluates models using:
- **MSE** (Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R²** (Coefficient of Determination)

## Advanced Features

### Custom Model Weights
```python
engine.ensemble.weights = {
    'lstm': 0.3,
    'xgboost': 0.3,
    'prophet': 0.2,
    'arima': 0.2
}
```

### Confidence Intervals
```python
conf_int = engine.ensemble.get_confidence_interval(
    data, 
    steps=5, 
    confidence=0.95
)
```

### Model Evaluation
```python
evaluation = engine.evaluate_models(data, test_size=0.2)
for model_name, metrics in evaluation.items():
    print(f"{model_name}: RMSE = {metrics['rmse']:.4f}")
```

## Troubleshooting

### Issue: Not enough data
- Ensure you have at least 60-100 data points
- Reduce `sequence_length` in config.yaml

### Issue: Web scraping fails
- Verify the website URL is accessible
- Check CSS selector is correct
- Review website's robots.txt

### Issue: Model training errors
- Check data quality (no NaN values)
- Ensure sufficient data points
- Try reducing model complexity

## Project Structure

```
prediction-system/
├── config.yaml              # Configuration file
├── requirements.txt         # Python dependencies
├── main.py                 # Main entry point
├── example_usage.py        # Usage examples
├── README.md               # This file
├── src/
│   ├── data_scraper.py     # Web scraping module
│   ├── prediction_engine.py # Main prediction engine
│   ├── ensemble_predictor.py # Ensemble methods
│   ├── visualizer.py       # Visualization tools
│   └── models/
│       ├── lstm_model.py   # LSTM implementation
│       ├── prophet_model.py # Prophet wrapper
│       ├── statistical_models.py # ARIMA, etc.
│       └── ml_models.py    # XGBoost, LightGBM
├── data/                   # Historical data storage
├── models/saved/           # Saved model files
└── outputs/                # Generated visualizations
```

## Requirements

- Python 3.8+
- TensorFlow 2.13+
- scikit-learn
- XGBoost
- LightGBM
- Prophet
- pandas, numpy
- matplotlib, seaborn, plotly
- BeautifulSoup4, requests

## License

This project is open source and available for personal and commercial use.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review example_usage.py for working examples
3. Verify configuration in config.yaml

## Next Steps

1. **Configure your data source** in config.yaml
2. **Run the quick example**: `python example_usage.py quick`
3. **Customize models** based on your data characteristics
4. **Evaluate performance** and adjust parameters
5. **Deploy** for production use

---

**Happy Predicting! 🚀**

---

## 🚀 Release Notes v4.0.1 (Tactical OS Deployment)

### 🛠️ Critical Deployment Fixes
- **Dependency Fix**: Added `python-dotenv` to `requirements.txt` to resolve 502 Bad Gateway errors on Dokploy/production servers.
- **Port Standardization**: Dockerfile now explicitly exposes port `8050`. Ensure your container orchestration maps this internal port correctly.
- **Layout Stabilization**: Fixed global horizontal scroll issues. Sidebar is now correctly pinned, and only the **History Strip** scrolls.

### ✨ New Features
- **Tactical OS V4**: Complete UI overhaul with "Glassmorphism" design and real-time telemetry.
- **Swahili Localization**: Full `SW` language support for all dashboard modules.
- **Live Crash History**: "NEW" badge indicator for the most recent round, with real-time left-side injection.
