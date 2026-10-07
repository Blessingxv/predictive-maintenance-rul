# Predictive Maintenance - Remaining Useful Life Prediction

A machine learning project for predicting the Remaining Useful Life (RUL) of turbofan engines using the NASA C-MAPSS dataset.

## Project Overview

The goal of this project is to develop and compare machine learning models capable of estimating how many operational cycles an engine has remaining before failure.

The project explores engine sensor data, identifies degradation patterns, and progressively develops models from simple baselines to neural networks.

## Dataset

This project uses the NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset.

Each engine contains a time series of operational cycles with:
- Engine/unit ID
- Operational settings
- Sensor measurements
- Remaining Useful Life (RUL) target derived from the final observed cycle

## Current Progress

### Exploratory Data Analysis
- Examined engine degradation trajectories
- Removed constant sensor features
- Analysed sensor-to-RUL correlations
- Analysed sensor-to-sensor correlations
- Examined the RUL distribution

### Data Preparation
- Created the RUL target
- Split training and validation data by engine ID to prevent data leakage
- Standardized sensor features using statistics learned only from the training set

### Baseline Models

A naive baseline that predicts the mean training RUL achieved:

- RMSE: 65.72 cycles

Linear Regression achieved:

- MAE: 30.10 cycles
- RMSE: 38.12 cycles
- R²: 0.663

Residual analysis showed systematic nonlinear prediction errors, motivating the use of nonlinear models.

## Next Steps

The next stage is to develop a neural network using PyTorch and compare its performance against the Linear Regression baseline.

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- scikit-learn
- PyTorch

## Project Structure

```text
predictive-maintenance-rul/
notebooks/
      01_eda.ipynb
  .gitignore
  README.md
  requirements.txt
