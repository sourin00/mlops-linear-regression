import numpy as np
import joblib
import os
from utils import quantize, dequantize, load_model


def main():
    """Main quantization function."""
    print("Loading trained model...")
    model = load_model("models/linear_regression_model.joblib")

    # Extract coefficients and intercept
    coef = model.coef_
    intercept = model.intercept_

    print(f"Original coefficients shape: {coef.shape}")
    print(f"Original intercept: {intercept}")
    print(f"Original coef values: {coef}")

    # Save raw parameters
    raw_params = {
        'coef': coef,
        'intercept': intercept
    }
    os.makedirs("models", exist_ok=True)
    joblib.dump(raw_params, "models/unquant_params.joblib")

    # 8-bit quantization (per-coefficient)
    quant_coef8, coef8_min, coef8_max = quantize(coef)
    quant_intercept8, int8_min, int8_max = quantize(np.array([intercept]))
    print(f"\nQuantizing intercept (8-bit, per-coefficient)...")
    print(f"Intercept value: {intercept:.8f}")
    quant_params = {
        'quant_coef8': quant_coef8,
        'coef8_min': coef8_min,
        'coef8_max': coef8_max,
        'quant_intercept8': quant_intercept8[0],
        'int8_min': int8_min[0],
        'int8_max': int8_max[0]
    }
    joblib.dump(quant_params, "models/quant_params.joblib", compress=3)
    print("Quantized parameters (8-bit, per-coefficient) saved to models/quant_params.joblib")

    quant_size8 = os.path.getsize("models/quant_params.joblib")
    orig_size = os.path.getsize("models/linear_regression_model.joblib")
    print(f"Model size before quantization: {orig_size/1024:.2f} KB")
    print(f"Model size after  quantization:  {quant_size8/1024:.2f} KB")
    print(f"Size reduction:                {(orig_size-quant_size8)/1024:.2f} KB")

    # Dequantize for inference (8-bit, per-coefficient)
    dequant_coef8 = dequantize(quant_coef8, coef8_min, coef8_max)
    dequant_intercept8 = dequantize(np.array([quant_params['quant_intercept8']]), np.array([quant_params['int8_min']]), np.array([quant_params['int8_max']]))[0]

    # Calculate coefficient and intercept errors
    coef_error8 = np.abs(coef - dequant_coef8).max()
    intercept_error8 = np.abs(intercept - dequant_intercept8)
    print(f"Max coefficient error: {coef_error8:.8f}")
    print(f"Intercept error: {intercept_error8:.8f}")

    # Inference test and output formatting
    from utils import load_dataset
    X_train, X_test, y_train, y_test = load_dataset()
    manual_dequant_pred8 = X_test[:5] @ dequant_coef8 + dequant_intercept8

    print("\nInference Test (first 5 samples):")
    print(f"Manual dequant predictions:     {manual_dequant_pred8}")

    original_vs_dequant_diff8 = np.abs(model.predict(X_test[:5]) - manual_dequant_pred8)
    print(f"Absolute differences: {original_vs_dequant_diff8}")
    print(f"Max difference: {original_vs_dequant_diff8.max()}")
    print(f"Mean difference: {original_vs_dequant_diff8.mean()}")
    max_diff8 = original_vs_dequant_diff8.max()
    if max_diff8 < 0.1:
        print(f"Quantization quality is good (max diff: {max_diff8:.6f})")
    elif max_diff8 < 1.0:
        print(f"Quantization quality is acceptable (max diff: {max_diff8:.6f})")
    else:
        print(f"Quantization quality is poor (max diff: {max_diff8:.6f})")

    # Calculate prediction errors for quantized (dequantized) model
    max_pred_error8 = np.max(np.abs(y_test - (X_test @ dequant_coef8 + dequant_intercept8)))
    mean_pred_error8 = np.mean(np.abs(y_test - (X_test @ dequant_coef8 + dequant_intercept8)))
    print(f"Max Prediction Error (quantized): {max_pred_error8:.4f}")
    print(f"Mean Prediction Error (quantized): {mean_pred_error8:.4f}")
    print("\nQuantization completed successfully!\n")

    # Calculate R2 and MSE for quantized model
    y_pred8 = X_test @ dequant_coef8 + dequant_intercept8
    from utils import calculate_metrics
    r2_8, mse_8 = calculate_metrics(y_test, y_pred8)
    print(f"R2 score (quantized model): {r2_8:.4f}")
    print(f"MSE (quantized model): {mse_8:.4f}")



if __name__ == "__main__":
    main()