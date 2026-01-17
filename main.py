import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import argparse
import logging
from dotenv import load_dotenv

from src.prediction_engine import PredictionEngine
from src.visualizer import PredictionVisualizer
from src.data_scraper import DataManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Advanced Prediction System')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--scrape', action='store_true',
                       help='Scrape data from website before prediction')
    parser.add_argument('--steps', type=int, default=1,
                       help='Number of steps to predict ahead')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations')
    parser.add_argument('--save-models', action='store_true',
                       help='Save trained models')
    parser.add_argument('--load-models', action='store_true',
                       help='Load previously trained models')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate model performance')
    parser.add_argument('--data-file', type=str,
                       help='Path to custom data file (CSV with "value" column)')
    
    args = parser.parse_args()
    
    load_dotenv()
    
    logger.info("=" * 60)
    logger.info("Advanced Prediction System - Starting")
    logger.info("=" * 60)
    
    engine = PredictionEngine(config_path=args.config)
    
    if args.scrape:
        logger.info("Scraping data from website...")
        engine.scrape_and_save_data()
    
    if args.data_file:
        import pandas as pd
        df = pd.read_csv(args.data_file)
        data = df['value'].values
        logger.info(f"Loaded {len(data)} data points from {args.data_file}")
    else:
        data = engine.load_data()
        logger.info(f"Loaded {len(data)} data points from database")
    
    logger.info("\nInitializing prediction models...")
    engine.initialize_models()
    
    if args.load_models:
        logger.info("Loading pre-trained models...")
        engine.load_models()
    else:
        logger.info("Training models on historical data...")
        engine.train_models(data)
    
    logger.info("\nGenerating predictions...")
    result = engine.predict(data, steps=args.steps)
    
    logger.info("\n" + "=" * 60)
    logger.info("PREDICTION RESULTS")
    logger.info("=" * 60)
    
    ensemble_pred = result['prediction']
    logger.info(f"\nEnsemble Prediction (next {args.steps} step(s)):")
    for i, pred in enumerate(ensemble_pred, 1):
        logger.info(f"  Step {i}: {pred:.4f}")
    
    conf_int = result['confidence_interval']
    logger.info(f"\nConfidence Interval ({conf_int['confidence']*100:.0f}%):")
    for i in range(len(ensemble_pred)):
        logger.info(f"  Step {i+1}: [{conf_int['lower_bound'][i]:.4f}, {conf_int['upper_bound'][i]:.4f}]")
    
    logger.info("\nIndividual Model Predictions:")
    for model_name, pred in result['individual_predictions'].items():
        logger.info(f"  {model_name:15s}: {pred[0]:.4f}")
    
    logger.info("\nModel Weights:")
    for model_name, weight in result['weights'].items():
        logger.info(f"  {model_name:15s}: {weight:.4f}")
    
    stats = result['statistics']
    logger.info("\nPrediction Statistics:")
    logger.info(f"  Mean:   {stats['mean'][0]:.4f}")
    logger.info(f"  Median: {stats['median'][0]:.4f}")
    logger.info(f"  Std:    {stats['std'][0]:.4f}")
    logger.info(f"  Range:  [{stats['min'][0]:.4f}, {stats['max'][0]:.4f}]")
    
    if args.evaluate:
        logger.info("\n" + "=" * 60)
        logger.info("MODEL EVALUATION")
        logger.info("=" * 60)
        evaluation = engine.evaluate_models(data, test_size=0.2)
        
        for model_name, metrics in evaluation.items():
            logger.info(f"\n{model_name}:")
            logger.info(f"  MSE:  {metrics['mse']:.6f}")
            logger.info(f"  MAE:  {metrics['mae']:.6f}")
            logger.info(f"  RMSE: {metrics['rmse']:.6f}")
            logger.info(f"  R²:   {metrics['r2']:.6f}")
    
    if args.save_models:
        logger.info("\nSaving trained models...")
        engine.save_models()
        logger.info("Models saved successfully")
    
    if args.visualize:
        logger.info("\nGenerating visualizations...")
        visualizer = PredictionVisualizer()
        
        os.makedirs('outputs', exist_ok=True)
        
        visualizer.plot_predictions(
            data[-100:],
            result,
            conf_int,
            save_path='outputs/predictions.png'
        )
        
        visualizer.plot_interactive_predictions(
            data[-100:],
            result,
            conf_int,
            save_path='outputs/predictions_interactive.html'
        )
        
        visualizer.plot_model_comparison(
            result,
            save_path='outputs/model_comparison.png'
        )
        
        logger.info("Visualizations saved to 'outputs/' directory")
    
    logger.info("\n" + "=" * 60)
    logger.info("Prediction Complete!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
