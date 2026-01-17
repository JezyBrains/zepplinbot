import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from src.prediction_engine import PredictionEngine
from src.visualizer import PredictionVisualizer


def example_with_sample_data():
    print("=" * 60)
    print("Example: Prediction with Sample Data")
    print("=" * 60)
    
    np.random.seed(42)
    trend = np.linspace(100, 200, 200)
    seasonal = 10 * np.sin(np.linspace(0, 8*np.pi, 200))
    noise = np.random.normal(0, 5, 200)
    sample_data = trend + seasonal + noise
    
    print(f"\nGenerated {len(sample_data)} sample data points")
    print(f"Data range: [{sample_data.min():.2f}, {sample_data.max():.2f}]")
    
    engine = PredictionEngine(config_path='config.yaml')
    
    print("\nInitializing and training models...")
    engine.initialize_models()
    engine.train_models(sample_data)
    
    print("\nMaking predictions for next 5 steps...")
    result = engine.predict(sample_data, steps=5)
    
    print("\n" + "-" * 60)
    print("RESULTS")
    print("-" * 60)
    
    print("\nEnsemble Predictions:")
    for i, pred in enumerate(result['prediction'], 1):
        print(f"  Step {i}: {pred:.2f}")
    
    print("\nConfidence Intervals (95%):")
    conf = result['confidence_interval']
    for i in range(len(result['prediction'])):
        print(f"  Step {i+1}: [{conf['lower_bound'][i]:.2f}, {conf['upper_bound'][i]:.2f}]")
    
    print("\nIndividual Model Predictions (Step 1):")
    for model_name, pred in result['individual_predictions'].items():
        print(f"  {model_name:15s}: {pred[0]:.2f}")
    
    visualizer = PredictionVisualizer()
    visualizer.plot_predictions(sample_data[-50:], result, conf)
    visualizer.plot_model_comparison(result)
    
    print("\n" + "=" * 60)
    print("Example Complete!")
    print("=" * 60)


def example_with_web_scraping():
    print("=" * 60)
    print("Example: Web Scraping and Prediction")
    print("=" * 60)
    print("\nNOTE: Update config.yaml with your website URL and selector")
    print("before running this example.\n")
    
    engine = PredictionEngine(config_path='config.yaml')
    
    print("Initializing web scraper...")
    engine.initialize_scraper()
    
    print("Scraping data from website...")
    df = engine.scrape_and_save_data()
    
    if df is not None and not df.empty:
        data = df['value'].values
        print(f"Scraped {len(data)} data points")
        
        engine.initialize_models()
        engine.train_models(data)
        
        result = engine.predict(data, steps=3)
        
        print("\nPredictions:")
        for i, pred in enumerate(result['prediction'], 1):
            print(f"  Step {i}: {pred:.2f}")
    else:
        print("No data scraped. Please check your configuration.")


def example_quick_prediction():
    print("=" * 60)
    print("Quick Prediction Example")
    print("=" * 60)
    
    data = np.array([10, 12, 15, 14, 18, 20, 22, 25, 23, 28, 
                     30, 32, 35, 33, 38, 40, 42, 45, 43, 48])
    
    print(f"\nInput data: {data}")
    print(f"Data points: {len(data)}")
    
    engine = PredictionEngine(config_path='config.yaml')
    engine.initialize_models()
    engine.train_models(data)
    
    result = engine.predict(data, steps=3)
    
    print("\nNext 3 predictions:")
    for i, pred in enumerate(result['prediction'], 1):
        print(f"  Step {i}: {pred:.2f}")
    
    print("\nModel contributions:")
    for model_name, weight in result['weights'].items():
        print(f"  {model_name}: {weight*100:.1f}%")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'sample':
            example_with_sample_data()
        elif sys.argv[1] == 'scrape':
            example_with_web_scraping()
        elif sys.argv[1] == 'quick':
            example_quick_prediction()
        else:
            print("Usage: python example_usage.py [sample|scrape|quick]")
    else:
        print("Running all examples...\n")
        example_quick_prediction()
        print("\n" * 2)
        example_with_sample_data()
