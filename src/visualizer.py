import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)
sns.set_style('whitegrid')


class PredictionVisualizer:
    def __init__(self, figsize: tuple = (14, 8)):
        self.figsize = figsize
        
    def plot_predictions(self, historical_data: np.ndarray, 
                        predictions: Dict, 
                        confidence_interval: Optional[Dict] = None,
                        save_path: Optional[str] = None):
        fig, ax = plt.subplots(figsize=self.figsize)
        
        historical_x = np.arange(len(historical_data))
        ax.plot(historical_x, historical_data, label='Historical Data', 
                color='blue', linewidth=2)
        
        pred_start = len(historical_data)
        ensemble_pred = predictions.get('prediction', predictions.get('ensemble'))
        pred_x = np.arange(pred_start, pred_start + len(ensemble_pred))
        
        ax.plot(pred_x, ensemble_pred, label='Ensemble Prediction', 
                color='red', linewidth=2, marker='o', markersize=8)
        
        if confidence_interval:
            lower = confidence_interval.get('lower_bound')
            upper = confidence_interval.get('upper_bound')
            if lower is not None and upper is not None:
                ax.fill_between(pred_x, lower, upper, alpha=0.3, color='red',
                               label=f'{confidence_interval.get("confidence", 0.95)*100:.0f}% Confidence Interval')
        
        if 'individual_predictions' in predictions:
            for model_name, pred in predictions['individual_predictions'].items():
                ax.plot(pred_x, pred, '--', alpha=0.5, label=f'{model_name}')
        
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title('Prediction Results', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
        return fig
    
    def plot_interactive_predictions(self, historical_data: np.ndarray,
                                    predictions: Dict,
                                    confidence_interval: Optional[Dict] = None,
                                    save_path: Optional[str] = None):
        fig = go.Figure()
        
        historical_x = list(range(len(historical_data)))
        fig.add_trace(go.Scatter(
            x=historical_x,
            y=historical_data,
            mode='lines',
            name='Historical Data',
            line=dict(color='blue', width=2)
        ))
        
        pred_start = len(historical_data)
        ensemble_pred = predictions.get('prediction', predictions.get('ensemble'))
        pred_x = list(range(pred_start, pred_start + len(ensemble_pred)))
        
        fig.add_trace(go.Scatter(
            x=pred_x,
            y=ensemble_pred,
            mode='lines+markers',
            name='Ensemble Prediction',
            line=dict(color='red', width=2),
            marker=dict(size=8)
        ))
        
        if confidence_interval:
            lower = confidence_interval.get('lower_bound')
            upper = confidence_interval.get('upper_bound')
            if lower is not None and upper is not None:
                fig.add_trace(go.Scatter(
                    x=pred_x + pred_x[::-1],
                    y=list(upper) + list(lower)[::-1],
                    fill='toself',
                    fillcolor='rgba(255,0,0,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'{confidence_interval.get("confidence", 0.95)*100:.0f}% CI',
                    showlegend=True
                ))
        
        if 'individual_predictions' in predictions:
            for model_name, pred in predictions['individual_predictions'].items():
                fig.add_trace(go.Scatter(
                    x=pred_x,
                    y=pred,
                    mode='lines',
                    name=model_name,
                    line=dict(dash='dash'),
                    opacity=0.6
                ))
        
        fig.update_layout(
            title='Interactive Prediction Results',
            xaxis_title='Time Step',
            yaxis_title='Value',
            hovermode='x unified',
            template='plotly_white',
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive plot saved to {save_path}")
        
        fig.show()
        return fig
    
    def plot_model_comparison(self, predictions: Dict, save_path: Optional[str] = None):
        if 'individual_predictions' not in predictions:
            logger.warning("No individual predictions to compare")
            return
        
        individual_preds = predictions['individual_predictions']
        weights = predictions.get('weights', {})
        
        models = list(individual_preds.keys())
        pred_values = [pred[0] if isinstance(pred, np.ndarray) else pred 
                      for pred in individual_preds.values()]
        model_weights = [weights.get(model, 0) for model in models]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
        bars = ax1.bar(models, pred_values, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Model', fontsize=12)
        ax1.set_ylabel('Predicted Value', fontsize=12)
        ax1.set_title('Model Predictions Comparison', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        
        for i, (bar, val) in enumerate(zip(bars, pred_values)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=10)
        
        if any(w > 0 for w in model_weights):
            wedges, texts, autotexts = ax2.pie(model_weights, labels=models, autopct='%1.1f%%',
                                               colors=colors, startangle=90)
            ax2.set_title('Model Weights in Ensemble', fontsize=14, fontweight='bold')
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {save_path}")
        
        plt.show()
        return fig
    
    def plot_error_analysis(self, actual: np.ndarray, predicted: np.ndarray,
                           save_path: Optional[str] = None):
        errors = actual - predicted
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        axes[0, 0].scatter(actual, predicted, alpha=0.6, edgecolors='k')
        axes[0, 0].plot([actual.min(), actual.max()], [actual.min(), actual.max()], 
                       'r--', lw=2, label='Perfect Prediction')
        axes[0, 0].set_xlabel('Actual Values', fontsize=11)
        axes[0, 0].set_ylabel('Predicted Values', fontsize=11)
        axes[0, 0].set_title('Actual vs Predicted', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].hist(errors, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(0, color='r', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Prediction Error', fontsize=11)
        axes[0, 1].set_ylabel('Frequency', fontsize=11)
        axes[0, 1].set_title('Error Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        axes[1, 0].plot(errors, marker='o', linestyle='-', alpha=0.6)
        axes[1, 0].axhline(0, color='r', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Sample Index', fontsize=11)
        axes[1, 0].set_ylabel('Prediction Error', fontsize=11)
        axes[1, 0].set_title('Error Over Time', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        from scipy import stats
        stats.probplot(errors, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Error analysis plot saved to {save_path}")
        
        plt.show()
        return fig
    
    def plot_time_series_decomposition(self, data: np.ndarray, period: int = 12,
                                      save_path: Optional[str] = None):
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        df = pd.DataFrame({'value': data})
        df.index = pd.date_range(start='2020-01-01', periods=len(data), freq='D')
        
        decomposition = seasonal_decompose(df['value'], model='additive', period=period)
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 10))
        
        decomposition.observed.plot(ax=axes[0], title='Original', color='blue')
        axes[0].grid(True, alpha=0.3)
        
        decomposition.trend.plot(ax=axes[1], title='Trend', color='green')
        axes[1].grid(True, alpha=0.3)
        
        decomposition.seasonal.plot(ax=axes[2], title='Seasonal', color='orange')
        axes[2].grid(True, alpha=0.3)
        
        decomposition.resid.plot(ax=axes[3], title='Residual', color='red')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Decomposition plot saved to {save_path}")
        
        plt.show()
        return fig
