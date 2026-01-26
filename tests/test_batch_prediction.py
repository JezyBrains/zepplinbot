import pytest
import numpy as np
import sys
import os

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from models.lstm_model import LSTMPredictor

def test_lstm_batch_prediction():
    # Setup
    np.random.seed(42)
    # Generate random data
    data = np.random.rand(200)
    sequence_length = 10

    # Initialize and train model
    # Use small units and epochs for speed
    model = LSTMPredictor(sequence_length=sequence_length, units=[16])

    # Train on first 100 points
    model.train(data[:100], epochs=1, batch_size=32)

    # Prepare test inputs
    start_idx = 100
    num_tests = 20

    # Create input sequences
    input_sequences = []

    for i in range(num_tests):
        # Taking sequences from raw data
        idx = start_idx + i
        # Ensure we have enough data
        if idx - sequence_length >= 0 and idx < len(data):
            seq = data[idx - sequence_length : idx]
            input_sequences.append(seq)

    input_sequences = np.array(input_sequences)

    assert len(input_sequences) == num_tests

    # Predict using iterative predict_next
    iterative_preds = []
    for seq in input_sequences:
        pred = model.predict_next(seq, steps=1)[0]
        iterative_preds.append(pred)

    iterative_preds = np.array(iterative_preds)

    # Predict using batch
    batch_preds = model.predict_batch(input_sequences)

    # Compare
    # Tolerance: Floating point differences can occur due to batching vs single inference in some frameworks,
    # but usually they are identical for CPU. Keras might have slight differences.
    np.testing.assert_allclose(iterative_preds, batch_preds, rtol=1e-5, atol=1e-5,
                               err_msg="Batch predictions do not match iterative predictions")
