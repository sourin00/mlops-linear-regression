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


def quantize_to_uint8(values, scale_factor=1000):
    """Quantize float values to uint8."""
    # Scale and clip values to fit in uint8 range
    scaled_values = values * scale_factor
    min_val, max_val = scaled_values.min(), scaled_values.max()

    # Normalize to 0-255 range
    normalized = ((scaled_values - min_val) / (max_val - min_val) * 255)
    quantized = normalized.astype(np.uint8)

    return quantized, min_val, max_val, scale_factor


def dequantize_from_uint8(quantized_values, min_val, max_val, scale_factor):
    """Dequantize uint8 values back to float."""
    # Denormalize from 0-255 range
    denormalized = (quantized_values.astype(np.float32) / 255.0) * (max_val - min_val) + min_val
    # Descale
    original_values = denormalized / scale_factor
    return original_values