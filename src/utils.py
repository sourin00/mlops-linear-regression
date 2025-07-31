import numpy as np
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import os


def load_dataset():
    """Load and split California Housing dataset."""
    data = fetch_california_housing()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def create_model():
    """Create LinearRegression model instance."""
    return LinearRegression()


def save_model(model, filepath):
    """Save model using joblib."""
    # Only create directory if filepath contains a directory
    directory = os.path.dirname(filepath)
    if directory:  # Only create directory if it's not empty
        os.makedirs(directory, exist_ok=True)
    joblib.dump(model, filepath)


def load_model(filepath):
    """Load model using joblib."""
    return joblib.load(filepath)


def calculate_metrics(y_true, y_pred):
    """Calculate R2 score and MSE."""
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return r2, mse

def quantize(values):
    """Per-channel quantize float values to uint8 using min-max linear scaling."""
    values = np.asarray(values)
    if values.ndim == 1:
        min_vals = values
        max_vals = values
        if values.size == 1:
            # Scalar case
            min_vals = max_vals = values
        else:
            min_vals = values.copy()
            max_vals = values.copy()
        # For 1D, treat as per-coefficient
        min_vals = values
        max_vals = values
    else:
        min_vals = values.min(axis=0)
        max_vals = values.max(axis=0)
    min_vals = np.asarray(min_vals)
    max_vals = np.asarray(max_vals)
    quantized = np.zeros_like(values, dtype=np.uint8)
    for i in range(values.size):
        v = values.flat[i]
        min_v = min_vals.flat[i]
        max_v = max_vals.flat[i]
        if max_v == min_v:
            quantized.flat[i] = 0
        else:
            scale = 255.0 / (max_v - min_v)
            quantized.flat[i] = np.round((v - min_v) * scale).astype(np.uint8)
    return quantized, min_vals.astype(float), max_vals.astype(float)


def dequantize(quantized, min_vals, max_vals):
    """Dequantize uint8 values back to float using per-channel min and max."""
    quantized = np.asarray(quantized, dtype=np.uint8)
    min_vals = np.asarray(min_vals)
    max_vals = np.asarray(max_vals)
    dequantized = np.zeros_like(quantized, dtype=np.float32)
    for i in range(quantized.size):
        min_v = min_vals.flat[i]
        max_v = max_vals.flat[i]
        if max_v == min_v:
            dequantized.flat[i] = min_v
        else:
            scale = (max_v - min_v) / 255.0
            dequantized.flat[i] = quantized.flat[i] * scale + min_v
    return dequantized
