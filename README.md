# MLOps Linear Regression Pipeline

A complete MLOps pipeline for Linear Regression using California Housing dataset with training, testing, quantization, Dockerization, and CI/CD automation.

## Table of Contents
- [Overview](#overview)
- [Repository Setup](#repository-setup)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Docker Usage](#docker-usage)
- [CI/CD Pipeline](#cicd-pipeline)
- [Testing](#testing)
- [Quantization Details](#quantization-details)

## Overview

This project implements a complete MLOps pipeline for Linear Regression using the California Housing dataset. The pipeline includes:

- **Model Training**: Linear Regression using scikit-learn
- **Testing**: Comprehensive unit tests with pytest
- **Quantization**: Manual quantization for model compression
- **Containerization**: Docker support for deployment
- **CI/CD**: Automated GitHub Actions workflow
- **Monitoring**: Performance tracking and comparison

## Repository Setup

This repository was created entirely using the Windows PowerShell command line tool. Here are the exact steps followed:

### Step 1: Create a Virtual Environment
```powershell
# Create a new virtual environment
python -m venv mlops_env

# Activate virtual environment (Windows PowerShell)
mlops_env\Scripts\Activate.ps1
```

### Step 2: Create Project Structure
```powershell
# Create project directory
New-Item -ItemType Directory -Name "mlops-linear-regression"
Set-Location "mlops-linear-regression"

# Create directory structure
New-Item -ItemType Directory -Path "src", "tests", ".github\workflows", "models" -Force

# Create initial files
New-Item -ItemType File -Name "README.md", ".gitignore", "requirements.txt"
New-Item -ItemType File -Path "src\__init__.py", "src\train.py", "src\quantize.py", "src\predict.py", "src\utils.py"
New-Item -ItemType File -Path "tests\__init__.py", "tests\test_train.py"
New-Item -ItemType File -Path ".github\workflows\ci.yml"
New-Item -ItemType File -Name "Dockerfile"
```

### Step 3: Create requirements.txt
```powershell
# Generate requirements file using PowerShell Here-String
@"
scikit-learn==1.3.0
numpy==1.24.3
joblib==1.3.2
pytest==7.4.0
pandas==2.0.3
"@ | Out-File -FilePath "requirements.txt" -Encoding UTF8
```

### Step 4: Install Dependencies
```powershell
# Install required packages
pip install -r requirements.txt

# Verify installation
pip list
```

### Step 5: Initialize Git Repository
```powershell
# Initialize git repository
git init

# Add all files to staging
git add .

# Create initial commit
git commit -m "Initial commit: MLOps Linear Regression Pipeline setup"

# Set master as default branch
git branch -M master
```

### Step 6: Connect to GitHub
```powershell
# Add remote origin (replace with your repository URL)
git remote add origin https://github.com/sourin00/mlops-linear-regression.git

# Push to GitHub
git push -u origin master
```

### Step 7: Verify Setup
```powershell
# Check git status
git status

# Check remote connection
git remote -v

# Check branch
git branch
```

## Project Structure
```
mlops-linear-regression/
├── .github/
│   └── workflows/
│       └── ci.yml
├── src/
│   ├── __init__.py
│   ├── train.py
│   ├── quantize.py
│   ├── predict.py
│   └── utils.py
├── tests/
│   ├── __init__.py
│   └── test_train.py
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation & Setup

### Prerequisites
- Python 3.9+ 
- Git
- Docker

### Local Setup
```powershell
# Clone the repository
git clone https://github.com/sourin00/mlops-linear-regression.git
cd mlops-linear-regression

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import sklearn, numpy, joblib, pytest; print('All dependencies installed successfully!')"
```

## Usage

### Training the Model
```powershell
# Navigate to source directory
cd src

# Train the linear regression model
python train.py
```

**Output:**
```
Loading California Housing dataset...
Creating LinearRegression model...
Training model...
R² Score: 0.5758
Mean Squared Error (Loss): 0.5559
Max Prediction Error: 9.8753
Mean Prediction Error: 0.5332
Model saved to models/linear_regression_model.joblib
```

### Running Quantization
```powershell
# Run model quantization
python quantize.py
```

**Output:**
```
Loading trained model...
Original coefficients shape: (8,)
Original intercept: -37.02327770606389
Original coef values: [ 4.48674910e-01  9.72425752e-03 -1.23323343e-01  7.83144907e-01
 -2.02962058e-06 -3.52631849e-03 -4.19792487e-01 -4.33708065e-01]

Quantizing intercept...
Intercept value: -37.02327771
Intercept scale factor: 1769.16
Quantized parameters saved to models/quant_params.joblib

Model size before quantization: 0.65 KB
Model size after quantization:  0.55 KB
Size reduction:                0.10 KB
Max coefficient error: 0.00001722
Intercept error: 0.00000000

Inference Test (first 5 samples):
Original predictions (sklearn): [0.71912284 1.76401657 2.70965883 2.83892593 2.60465725]
Manual original predictions:    [0.71912284 1.76401657 2.70965883 2.83892593 2.60465725]
Manual dequant predictions:     [0.69951698 1.74201847 2.69089815 2.81510208 2.5894276 ]

Differences:
Sklearn vs manual original: [0. 0. 0. 0. 0.]
Original vs dequant manual:  [0.01960586 0.0219981  0.01876068 0.02382385 0.01522965]
Absolute differences: [0.01960586 0.0219981  0.01876068 0.02382385 0.01522965]
Max difference: 0.02382384825320827
Mean difference: 0.01988362812321611
Quantization quality is good (max diff: 0.023824)
Max Prediction Error (quantized): 9.8738
Mean Prediction Error (quantized): 0.5305
```

### Making Predictions
```powershell
# Run prediction script
python predict.py
```

## Model Performance

### Performance Comparison Table

| Metric                    | Original Model | Quantized Model (16-bit) | Quantized Model (8-bit) | 16-bit vs Original | 8-bit vs Original |
|---------------------------|----------------|--------------------------|------------------------|--------------------|-------------------|
| **R² Score**              | 0.5758         | 0.5752                   | -46.6831               | -0.0006            | -47.2589          |
| **MSE**                   | 0.5559         | 0.5567                   | 62.4844                | +0.0008            | +61.9285          |
| **Max Prediction Error**  | 9.8753         | 9.8738                   | 68.9402                | -0.0015            | +59.0649          |
| **Mean Prediction Error** | 0.5332         | 0.5305                   | 6.3301                 | -0.0027            | +5.7969           |
| **Model Size**            | 0.65 KB        | 0.55 KB                  | 0.53 KB                | -0.10 KB           | -0.12 KB          |


#### Why use 16-bit quantization instead of 8-bit?
While 8-bit quantization offers greater model size reduction, it can lead to a more significant drop in model accuracy (R², MSE, and prediction error) compared to 16-bit quantization. 16-bit quantization provides a better balance between compression and prediction quality, making it preferable for scenarios where accuracy is important but some size reduction is still desired.

## Docker Usage

### Building the Container
```powershell
# Build Docker image
docker build -t mlops-linear-regression .

# Verify image creation
docker images | Where-Object {$_.Repository -eq "mlops-linear-regression"}
```

### Running the Container
```powershell
# Run prediction container
docker run --rm mlops-linear-regression

# Run with interactive mode
docker run -it --rm mlops-linear-regression cmd
```

### Container Output
```
Loading trained model...
Loading test dataset...
Making predictions...
Model Performance:
R² Score: 0.5758
Mean Squared Error: 0.5559

Sample Predictions (first 10):
True: 0.48 | Predicted: 0.72 | Diff: 0.24
True: 0.46 | Predicted: 1.76 | Diff: 1.31
True: 5.00 | Predicted: 2.71 | Diff: 2.29
True: 2.19 | Predicted: 2.84 | Diff: 0.65
True: 2.78 | Predicted: 2.60 | Diff: 0.18
True: 1.59 | Predicted: 2.01 | Diff: 0.42
True: 1.98 | Predicted: 2.65 | Diff: 0.66
True: 1.57 | Predicted: 2.17 | Diff: 0.59
True: 3.40 | Predicted: 2.74 | Diff: 0.66
True: 4.47 | Predicted: 3.92 | Diff: 0.55

Prediction completed successfully!
```

## CI/CD Pipeline

The GitHub Actions workflow automatically runs on every push to master branch:

### Pipeline Stages

1. **Test Suite**
   - Runs comprehensive pytest suite
   - Validates data loading, model creation, training, and performance
   - Must pass before other jobs execute

2. **Train and Quantize**
   - Trains Linear Regression model
   - Performs quantization
   - Uploads model artifacts for next stage

3. **Build and Test Container**
   - Downloads trained model artifacts
   - Builds Docker image
   - Tests container execution with predict.py

### Workflow Status
```yaml
test_suite → train_and_quantize → build_and_test_container
```

### Viewing Pipeline Results
```bash
# Check workflow status
git push origin master

# Monitor at: https://github.com/sourin00/mlops-linear-regression/actions
```

## Testing

### Running Tests Locally
```bash
# Run all tests
cd src
python -m pytest ../tests/ -v

# Run specific test file
python -m pytest ../tests/test_train.py -v

# Run with coverage
python -m pytest ../tests/ --cov=. --cov-report=html
```

### Test Coverage
Our test suite covers:
- **Dataset Loading**: Validates California Housing data structure
- **Model Creation**: Ensures LinearRegression instantiation
- **Model Training**: Verifies coefficient existence and training success
- **Performance Threshold**: Confirms R² score > 0.5
- **Model Persistence**: Tests save/load functionality

### Expected Test Output
```
tests/test_train.py::TestTraining::test_dataset_loading PASSED
tests/test_train.py::TestTraining::test_model_creation PASSED
tests/test_train.py::TestTraining::test_model_training PASSED
tests/test_train.py::TestTraining::test_model_performance PASSED
tests/test_train.py::TestTraining::test_model_save_load PASSED

======== 5 passed in 2.34s ========
```

## Quantization Details

### Manual Quantization Process

Our custom quantization implementation:

1. **Parameter Extraction**: Extract `coef_` and `intercept_` from trained model
2. **Scaling**: Apply scale factor to fit float range into uint16
3. **Normalization**: Map values to 0-65535 range
4. **Storage**: Save quantized parameters with metadata
5. **Dequantization**: Reverse process for inference

### Quantization Formula
```python
# Quantization
scaled_values = values * scale_factor
normalized = ((scaled_values - min_val) / (max_val - min_val)) * 65535
quantized = normalized.astype(np.uint16)

# Dequantization  
denormalized = (quantized / 65535.0) * (max_val - min_val) + min_val
original = denormalized / scale_factor
```