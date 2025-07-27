import numpy as np
import joblib
import os
from utils import load_model, quantize_to_uint8, dequantize_from_uint8


def main():
    """Main quantization function."""
    print("Loading trained model...")
    model = load_model("models/linear_regression_model.joblib")

    # Extract coefficients and intercept
    coef = model.coef_
    intercept = model.intercept_

    print(f"Original coefficients shape: {coef.shape}")
    print(f"Original intercept: {intercept}")

    # Save raw parameters
    raw_params = {
        'coef': coef,
        'intercept': intercept
    }
    os.makedirs("models", exist_ok=True)
    joblib.dump(raw_params, "models/unquant_params.joblib")
    print("Raw parameters saved to models/unquant_params.joblib")

    # Quantize coefficients
    quant_coef, coef_min, coef_max, coef_scale = quantize_to_uint8(coef)

    # Quantize intercept (reshape to array for consistent processing)
    intercept_array = np.array([intercept])
    quant_intercept, int_min, int_max, int_scale = quantize_to_uint8(intercept_array)

    # Save quantized parameters with metadata
    quant_params = {
        'quant_coef': quant_coef,
        'quant_intercept': quant_intercept,
        'coef_min': coef_min,
        'coef_max': coef_max,
        'coef_scale': coef_scale,
        'int_min': int_min,
        'int_max': int_max,
        'int_scale': int_scale
    }
    joblib.dump(quant_params, "models/quant_params.joblib")
    print("Quantized parameters saved to models/quant_params.joblib")

    # Test dequantization
    dequant_coef = dequantize_from_uint8(quant_coef, coef_min, coef_max, coef_scale)
    dequant_intercept = dequantize_from_uint8(quant_intercept, int_min, int_max, int_scale)[0]

    print("\nQuantization Results:")
    print(f"Original coef range: [{coef.min():.6f}, {coef.max():.6f}]")
    print(f"Dequantized coef range: [{dequant_coef.min():.6f}, {dequant_coef.max():.6f}]")
    print(f"Original intercept: {intercept:.6f}")
    print(f"Dequantized intercept: {dequant_intercept:.6f}")

    # Test inference with dequantized weights
    from utils import load_dataset
    X_train, X_test, y_train, y_test = load_dataset()

    # Original predictions
    original_pred = model.predict(X_test[:5])

    # Quantized predictions
    quantized_pred = X_test[:5] @ dequant_coef + dequant_intercept

    print("\nInference Test (first 5 samples):")
    print("Original predictions:", original_pred)
    print("Quantized predictions:", quantized_pred)
    print("Max difference:", np.abs(original_pred - quantized_pred).max())


if __name__ == "__main__":
    main()