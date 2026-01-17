#!/usr/bin/env python3
"""
Streaming Data Pipeline

Real-time data ingestion with sliding window buffer,
event-driven callbacks, and live model updates.
"""

import numpy as np
import pandas as pd
from typing import Callable, List, Optional, Dict, Any
from collections import deque
from datetime import datetime
import threading
import time
import logging
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataBuffer:
    """
    Thread-safe sliding window data buffer.
    
    Maintains a fixed-size window of recent data points
    for real-time analysis.
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._buffer: deque = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._timestamps: deque = deque(maxlen=max_size)
    
    def add(self, value: float, timestamp: Optional[datetime] = None):
        """Add a new data point to the buffer."""
        with self._lock:
            self._buffer.append(value)
            self._timestamps.append(timestamp or datetime.now())
    
    def get_array(self) -> np.ndarray:
        """Get buffer contents as numpy array."""
        with self._lock:
            return np.array(list(self._buffer))
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get buffer contents as DataFrame."""
        with self._lock:
            return pd.DataFrame({
                'timestamp': list(self._timestamps),
                'value': list(self._buffer)
            })
    
    def get_last(self, n: int = 1) -> np.ndarray:
        """Get last n values."""
        with self._lock:
            data = list(self._buffer)
            return np.array(data[-n:])
    
    def __len__(self) -> int:
        return len(self._buffer)
    
    def clear(self):
        """Clear the buffer."""
        with self._lock:
            self._buffer.clear()
            self._timestamps.clear()


class EventEmitter:
    """Simple event emitter for callbacks."""
    
    def __init__(self):
        self._listeners: Dict[str, List[Callable]] = {}
    
    def on(self, event: str, callback: Callable):
        """Register a callback for an event."""
        if event not in self._listeners:
            self._listeners[event] = []
        self._listeners[event].append(callback)
    
    def off(self, event: str, callback: Callable):
        """Remove a callback."""
        if event in self._listeners:
            self._listeners[event].remove(callback)
    
    def emit(self, event: str, *args, **kwargs):
        """Emit an event to all listeners."""
        if event in self._listeners:
            for callback in self._listeners[event]:
                try:
                    callback(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in callback for {event}: {e}")


class StreamingPipeline:
    """
    Real-time data ingestion and processing pipeline.
    
    Features:
    - Sliding window data buffer
    - Event-driven architecture
    - CSV persistence
    - Model update triggers
    """
    
    def __init__(self,
                 buffer_size: int = 1000,
                 csv_path: Optional[str] = None,
                 auto_save: bool = True,
                 save_interval: int = 10):
        """
        Args:
            buffer_size: Maximum buffer size
            csv_path: Path to CSV file for persistence
            auto_save: Whether to auto-save to CSV
            save_interval: Save every N data points
        """
        self.buffer = DataBuffer(max_size=buffer_size)
        self.events = EventEmitter()
        self.csv_path = csv_path
        self.auto_save = auto_save
        self.save_interval = save_interval
        self._data_count = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Load existing data if CSV exists
        if csv_path and os.path.exists(csv_path):
            self._load_from_csv()
    
    def _load_from_csv(self):
        """Load existing data from CSV."""
        try:
            df = pd.read_csv(self.csv_path)
            for _, row in df.iterrows():
                ts = pd.to_datetime(row['timestamp']) if 'timestamp' in row else None
                self.buffer.add(row['value'], ts)
            logger.info(f"Loaded {len(df)} records from {self.csv_path}")
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
    
    def ingest(self, value: float, timestamp: Optional[datetime] = None, metadata: Dict = None):
        """
        Ingest a new data point.
        
        Triggers:
        - 'data': On every new data point
        - 'batch': Every save_interval points
        - 'threshold': When value crosses thresholds
        """
        timestamp = timestamp or datetime.now()
        self.buffer.add(value, timestamp)
        self._data_count += 1
        
        # Emit data event
        self.events.emit('data', value, timestamp, metadata)
        
        # Check for batch event
        if self._data_count % self.save_interval == 0:
            self.events.emit('batch', self.buffer.get_last(self.save_interval))
            
            if self.auto_save and self.csv_path:
                self._append_to_csv(value, timestamp)
        
        # Check thresholds
        self._check_thresholds(value)
    
    def _append_to_csv(self, value: float, timestamp: datetime):
        """Append data to CSV file."""
        try:
            file_exists = os.path.exists(self.csv_path)
            with open(self.csv_path, 'a') as f:
                if not file_exists:
                    f.write("timestamp,value\n")
                f.write(f"{timestamp.isoformat()},{value}\n")
        except Exception as e:
            logger.error(f"Error saving to CSV: {e}")
    
    def _check_thresholds(self, value: float):
        """Check if value crosses any thresholds."""
        thresholds = [1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
        for threshold in thresholds:
            if value >= threshold:
                self.events.emit('threshold', threshold, value)
    
    def on_data(self, callback: Callable[[float, datetime, Dict], None]):
        """Register callback for each data point."""
        self.events.on('data', callback)
    
    def on_batch(self, callback: Callable[[np.ndarray], None]):
        """Register callback for batch updates."""
        self.events.on('batch', callback)
    
    def on_threshold(self, callback: Callable[[float, float], None]):
        """Register callback for threshold crossings."""
        self.events.on('threshold', callback)
    
    def get_statistics(self) -> Dict:
        """Get current buffer statistics."""
        data = self.buffer.get_array()
        if len(data) == 0:
            return {'count': 0}
        
        return {
            'count': len(data),
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'median': float(np.median(data)),
            'last': float(data[-1]),
            'last_10_mean': float(np.mean(data[-10:])) if len(data) >= 10 else None
        }
    
    def start_simulation(self, 
                        data_source: pd.DataFrame,
                        speed: float = 1.0,
                        loop: bool = False):
        """
        Start simulated streaming from DataFrame.
        
        Args:
            data_source: DataFrame with 'value' column
            speed: Multiplier for playback speed
            loop: Whether to loop the data
        """
        def _stream():
            idx = 0
            values = data_source['value'].values
            
            while self._running:
                value = values[idx]
                self.ingest(value)
                
                idx += 1
                if idx >= len(values):
                    if loop:
                        idx = 0
                    else:
                        break
                
                time.sleep(1.0 / speed)
        
        self._running = True
        self._thread = threading.Thread(target=_stream, daemon=True)
        self._thread.start()
        logger.info("Started streaming simulation")
    
    def stop(self):
        """Stop streaming."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("Stopped streaming")


class ModelUpdateTrigger:
    """
    Triggers model updates based on streaming data conditions.
    """
    
    def __init__(self, 
                 pipeline: StreamingPipeline,
                 update_interval: int = 50,
                 drift_threshold: float = 0.2):
        """
        Args:
            pipeline: StreamingPipeline instance
            update_interval: Trigger update every N points
            drift_threshold: Trigger if mean shifts by this fraction
        """
        self.pipeline = pipeline
        self.update_interval = update_interval
        self.drift_threshold = drift_threshold
        self._baseline_mean: Optional[float] = None
        self._callbacks: List[Callable] = []
        self._count = 0
        
        # Register with pipeline
        pipeline.on_data(self._on_data)
    
    def on_update_needed(self, callback: Callable[[str, Dict], None]):
        """Register callback for when update is needed."""
        self._callbacks.append(callback)
    
    def _on_data(self, value: float, timestamp: datetime, metadata: Dict):
        """Handle incoming data."""
        self._count += 1
        
        # Interval-based update
        if self._count % self.update_interval == 0:
            self._trigger_update("interval", {
                'count': self._count,
                'reason': f'Every {self.update_interval} points'
            })
        
        # Drift-based update
        data = self.pipeline.buffer.get_array()
        if len(data) >= 100:
            if self._baseline_mean is None:
                self._baseline_mean = np.mean(data[:50])
            
            recent_mean = np.mean(data[-50:])
            drift = abs(recent_mean - self._baseline_mean) / (self._baseline_mean + 1e-10)
            
            if drift > self.drift_threshold:
                self._trigger_update("drift", {
                    'baseline_mean': self._baseline_mean,
                    'recent_mean': recent_mean,
                    'drift': drift
                })
                self._baseline_mean = recent_mean
    
    def _trigger_update(self, reason: str, details: Dict):
        """Trigger model update callbacks."""
        for callback in self._callbacks:
            try:
                callback(reason, details)
            except Exception as e:
                logger.error(f"Error in update callback: {e}")


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("STREAMING PIPELINE DEMO")
    print("=" * 60)
    
    # Create pipeline
    pipeline = StreamingPipeline(
        buffer_size=500,
        csv_path='data/stream_test.csv',
        auto_save=True,
        save_interval=5
    )
    
    # Register callbacks
    def on_data(value, ts, meta):
        print(f"[DATA] {ts.strftime('%H:%M:%S')} - Value: {value:.2f}")
    
    def on_batch(batch):
        print(f"[BATCH] Received {len(batch)} points, mean: {np.mean(batch):.2f}")
    
    def on_threshold(threshold, value):
        print(f"[THRESHOLD] Value {value:.2f} crossed {threshold}x")
    
    pipeline.on_data(on_data)
    pipeline.on_batch(on_batch)
    pipeline.on_threshold(on_threshold)
    
    # Model update trigger
    trigger = ModelUpdateTrigger(pipeline, update_interval=10)
    
    def on_model_update(reason, details):
        print(f"[MODEL UPDATE] Reason: {reason}, Details: {details}")
    
    trigger.on_update_needed(on_model_update)
    
    # Load sample data and simulate
    df = pd.read_csv('data/zeppelin_data.csv')
    
    print("\nStarting simulation (5 seconds)...\n")
    pipeline.start_simulation(df, speed=5.0)
    
    time.sleep(5)
    
    pipeline.stop()
    
    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)
    print(json.dumps(pipeline.get_statistics(), indent=2))
