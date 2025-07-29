import numpy as np
import joblib
import os

from utils import quantize_to_uint16, dequantize_from_uint16, load_model


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

    quant_coef16, coef16_min, coef16_max, coef16_scale = quantize_to_uint16(coef)
    quant_intercept16, int16_min, int16_max, int16_scale = quantize_to_uint16(np.array([intercept]))

    print(f"\nQuantizing intercept...")
    print(f"Intercept value: {intercept:.8f}")
    print(f"Intercept scale factor: {int16_scale:.2f}")

    # Save quantized parameters with metadata
    quant_params = {
        'quant_coef16': quant_coef16,
        'coef16_min': coef16_min,
        'coef16_max': coef16_max,
        'coef16_scale': coef16_scale,
        'quant_intercept16': quant_intercept16[0],
        'int16_min': int16_min,
        'int16_max': int16_max,
        'int16_scale': int16_scale
    }
    joblib.dump(quant_params, "models/quant_params.joblib")
    print("Quantized parameters saved to models/quant_params.joblib")

    # Print model size difference

    orig_size = os.path.getsize("models/linear_regression_model.joblib")
    quant_size = os.path.getsize("models/quant_params.joblib")
    print(f"\nModel size before quantization: {orig_size/1024:.2f} KB")
    print(f"Model size after quantization:  {quant_size/1024:.2f} KB")
    print(f"Size reduction:                {(orig_size-quant_size)/1024:.2f} KB")

    # Dequantize for inference
    dequant_coef16 = dequantize_from_uint16(quant_coef16, coef16_min, coef16_max, coef16_scale)
    dequant_intercept16 = dequantize_from_uint16(np.array([quant_params['quant_intercept16']]), int16_min, int16_max, int16_scale)[0]

    # Calculate coefficient and intercept errors
    coef_error = np.abs(coef - dequant_coef16).max()
    intercept_error = np.abs(intercept - dequant_intercept16)
    print(f"Max coefficient error: {coef_error:.8f}")
    print(f"Intercept error: {intercept_error:.8f}")

    # Inference test and output formatting
    from utils import load_dataset
    X_train, X_test, y_train, y_test = load_dataset()

    # First 5 samples
    original_pred = model.predict(X_test[:5])
    manual_original_pred = X_test[:5] @ coef + intercept
    manual_dequant_pred = X_test[:5] @ dequant_coef16 + dequant_intercept16

    print("\nInference Test (first 5 samples):")
    print(f"Original predictions (sklearn): {original_pred}")
    print(f"Manual original predictions:    {manual_original_pred}")
    print(f"Manual dequant predictions:     {manual_dequant_pred}")

    print("\nDifferences:")
    print(f"Sklearn vs manual original: {np.abs(original_pred - manual_original_pred)}")
    print(f"Original vs dequant manual:  {np.abs(manual_original_pred - manual_dequant_pred)}")
    original_vs_dequant_diff = np.abs(original_pred - manual_dequant_pred)
    print(f"Absolute differences: {original_vs_dequant_diff}")
    print(f"Max difference: {original_vs_dequant_diff.max()}")
    print(f"Mean difference: {original_vs_dequant_diff.mean()}")
    max_diff = original_vs_dequant_diff.max()
    if max_diff < 0.1:
        print(f"Quantization quality is good (max diff: {max_diff:.6f})")
    elif max_diff < 1.0:
        print(f"Quantization quality is acceptable (max diff: {max_diff:.6f})")
    else:
        print(f"Quantization quality is poor (max diff: {max_diff:.6f})")

    # Calculate prediction errors for quantized (dequantized) model
    max_pred_error = np.max(np.abs(y_test - (X_test @ dequant_coef16 + dequant_intercept16)))
    mean_pred_error = np.mean(np.abs(y_test - (X_test @ dequant_coef16 + dequant_intercept16)))
    print(f"Max Prediction Error (quantized): {max_pred_error:.4f}")
    print(f"Mean Prediction Error (quantized): {mean_pred_error:.4f}")


if __name__ == "__main__":
    main()