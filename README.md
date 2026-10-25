# Predictive Maintenance: Remaining Useful Life (RUL) Prediction

A machine learning project that predicts the **Remaining Useful Life (RUL)** of turbofan engines using sensor measurements and a neural network built with **PyTorch**.

The project uses NASA's **C-MAPSS FD001 dataset** to explore predictive maintenance, feature engineering, model evaluation, and the challenges of estimating equipment failure.

**Project status:** Machine learning v1 completed and evaluated. Further model improvements and development of a FastAPI backend and web application are planned.

## Project Overview

Predictive maintenance aims to identify potential equipment failures before they occur, allowing maintenance teams to make better-informed decisions.

This project estimates the number of operating cycles remaining before an aircraft engine reaches failure.

The current implementation includes:

- Exploratory data analysis and sensor selection
- Remaining Useful Life target construction
- Baseline models for comparison
- A feedforward neural network built with PyTorch
- Rolling-average feature engineering
- Controlled experiments across multiple random seeds
- Evaluation using the official NASA FD001 test dataset
- Saving the trained model and preprocessing artifacts

The longer-term goal is to turn this machine learning project into a software application that allows users to submit engine sensor measurements and receive RUL predictions.

## Key Results

The final neural network was trained on all 100 FD001 training engines and evaluated on the official test dataset, using the last available observation from each of 100 test engines.

| Metric | Result |
|---|---:|
| Test RMSE | **32.90 cycles** |
| Test MAE | **24.14 cycles** |
| Mean signed error | +20.50 cycles |
| Overpredictions | 86 / 100 engines |
| Underpredictions | 14 / 100 engines |
| Minimum predicted RUL | 9.95 cycles |
| Maximum predicted RUL | 189.01 cycles |
| Maximum actual test RUL | 145 cycles |

**Important finding:** The model overpredicts the remaining useful life of most test engines. In predictive maintenance, overprediction can be particularly concerning because it may suggest that equipment can operate longer than it safely should.

These results establish a working machine learning baseline, **not a model ready for real-world maintenance decisions**.

## Dataset

**Dataset:** NASA C-MAPSS - FD001

The dataset contains simulated degradation measurements from turbofan engines.

| Property | Description |
|---|---|
| Training engines | 100 |
| Training observations | 20,631 |
| Test engines | 100 |
| Operating conditions | 1 |
| Fault modes | 1 |
| Sensor measurements | 21 original sensors |

Training engines contain run-to-failure trajectories, while test trajectories end before failure.

The remaining useful life for each training observation is calculated as:

```python
RUL = max_cycle_for_engine - current_cycle
```

The training targets are currently **uncapped**. Investigating capped RUL targets is a planned model improvement.

## Methodology

### 1. Exploratory Data Analysis

The dataset was explored to understand engine operating cycles, sensor behaviour, degradation patterns, and relationships between sensor measurements and RUL.

Constant sensor columns were excluded from modelling.

### 2. Feature Engineering

The initial neural network used 15 non-constant sensor features:

`2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21`

Two additional features were introduced:

- `sensor_11_rolling_mean`
- `sensor_12_rolling_mean`

These represent trailing **5-cycle rolling averages**, calculated separately for each engine.

The rolling windows use `min_periods=1`, allowing averages to be calculated during the first four operating cycles using the observations available at that point.

Before calculating rolling averages, observations are sorted by engine ID and operating cycle. This preserves chronological order and prevents sensor histories from different engines being mixed.

The final model therefore uses **17 input features**.

### 3. Data Preprocessing

Features are standardized using `StandardScaler`.

During development, the dataset is split by **engine ID**, rather than by individual observations, to prevent the same engine appearing in both training and validation.

The scaler is fitted only on the training partition and then applied to validation data.

For final training, a new scaler is fitted using all 100 training engines.

The test preprocessing pipeline follows this order:

1. Sort observations by engine and operating cycle.
2. Calculate rolling-average features using each engine's available history.
3. Select the final recorded observation for each test engine.
4. Select the 17 features in the saved feature order.
5. Apply the fitted scaler and generate predictions.

### 4. Neural Network

The model is a fully connected feedforward neural network.

```text
Input: 17 features
    ↓
Linear layer: 64 neurons
    ↓
ReLU
    ↓
Linear layer: 32 neurons
    ↓
ReLU
    ↓
Output: 1 predicted RUL value
```

**Training configuration**

| Parameter | Value |
|---|---|
| Framework | PyTorch |
| Loss function | Mean Squared Error |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 64 |
| Final training epochs | 10 |

The project also contains an earlier 15-feature neural network experiment in `02_neural_network.ipynb`, which explored different training durations.

## Model Comparison

During development, models were evaluated on a held-out validation set containing 20 engines.

| Model | Validation RMSE |
|---|---:|
| Mean prediction baseline | 65.72 |
| Linear regression | 38.12 |
| Neural network — 15 features | 34.87 ± 0.25 |
| Neural network — 17 features | **34.28 ± 0.13** |

The neural network comparisons were repeated using five random seeds under the same training configuration.

Adding rolling-average features improved average validation RMSE by approximately **0.59 cycles**, with the 17-feature model performing better in all five runs.

However, this is a modest improvement measured using one engine-level validation split. Different random seeds measure training variability, not variability across different validation splits.

**Validation and test RMSE are not directly comparable:** validation evaluates all observations from 20 engines, whereas official test evaluation uses one final observation per engine.

## Error Analysis

The official test evaluation produced:

- **MAE:** 24.14 cycles
- **RMSE:** 32.90 cycles
- **Mean signed error:** +20.50 cycles

The positive mean signed error indicates a tendency to predict RUL values above the actual values.

One possible explanation is a mismatch between the distribution of uncapped training RUL values and the remaining lifetimes represented by the test snapshots. The model is trained on observations across complete engine lifetimes, including early-life observations with high RUL values.

However, this explanation remains a **hypothesis**, not a confirmed cause.

Future experiments will investigate RUL capping, error patterns across different RUL ranges, and whether these changes reduce overprediction.

## Limitations

The current implementation has several limitations:

- **Overprediction bias:** 86 of 100 test engines received RUL predictions above their actual remaining lifetimes.
- **Uncapped RUL targets:** The effect of introducing an RUL cap has not yet been tested.
- **Limited validation:** The five-seed comparison uses one fixed split of 20 validation engines.
- **Single final training run:** The official test result represents one trained final model.
- **No test-snapshot baseline:** The mean and linear regression baselines have not yet been evaluated on the same 100 test snapshots.
- **Limited temporal context:** The model uses individual observations supplemented by two rolling-average features rather than a complete sequence model.
- **Dataset scope:** FD001 represents one simulated operating condition and fault mode.
- **Evaluation metrics:** RMSE and MAE do not distinguish between the consequences of overpredicting and underpredicting RUL.

The model has not been validated for use on real industrial equipment.

## Next Phase: Model Improvements and Software Engineering

The next stage of this project will extend the existing machine learning work into a more complete software system.

### Machine Learning Improvements

Planned experiments include:

- Investigating capped versus uncapped RUL targets
- Evaluating baseline models on the official test snapshots
- Using engine-grouped cross-validation to assess generalization
- Investigating overprediction across different RUL ranges
- Exploring sequence-based architectures such as LSTMs
- Implementing the asymmetric C-MAPSS scoring function as an additional evaluation metric

### Backend and Web Application

The software engineering phase will focus on deploying the trained model and making predictions accessible through an application.

**Planned architecture:**

- **FastAPI backend:** Expose model inference through REST API endpoints.
- **Inference pipeline:** Load the saved model, scaler, and feature configuration.
- **Input validation:** Validate submitted sensor measurements and required engine history.
- **Web interface:** Allow users to submit sensor data and view predicted remaining useful life.
- **Prediction visualization:** Display model outputs and relevant sensor information.
- **Testing and deployment:** Add automated tests, containerization, and deployment.

Because the current model uses rolling-average features, inference will need access to the relevant recent sensor history rather than only a single isolated measurement.

**The API and web application have not yet been implemented.**

## Repository Structure

```text
predictive-maintenance-rul/
├── artifacts/
│   ├── rul_model.pth
│   ├── rul_scaler.pkl
│   └── feature_columns.json
├── data/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_neural_network.ipynb
│   └── 03_feature_engineering.ipynb
├── requirements.txt
└── README.md
```

**Notebook descriptions**

- `01_eda.ipynb` — Data exploration, RUL construction, and baseline models.
- `02_neural_network.ipynb` — Initial PyTorch neural network experiments.
- `03_feature_engineering.ipynb` — Rolling features, controlled model comparisons, final training, official test evaluation, and artifact persistence.

**Saved artifacts**

- `rul_model.pth` — Trained neural network weights.
- `rul_scaler.pkl` — Fitted feature scaler.
- `feature_columns.json` — Feature names in the expected input order.

## Installation and Usage

Clone the repository:

```bash
git clone https://github.com/Blessingxv/predictive-maintenance-rul.git
cd predictive-maintenance-rul
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the FD001 dataset from the [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).

Place the following files inside `data/`, using the paths expected by the notebooks:

```text
train_FD001.txt
test_FD001.txt
RUL_FD001.txt
```

Run the notebooks in numerical order using Jupyter Notebook or VS Code with Jupyter support.

## Technologies Used

- Python
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook
- Git and GitHub

**Planned:** FastAPI, web frontend, Docker, and deployment tooling.

## Acknowledgements

The dataset used in this project was developed by NASA and made available through the Prognostics Center of Excellence.

**Dataset:** NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS), FD001.

---

**Current milestone:** Machine learning baseline completed, evaluated, and saved.

**Next milestone:** Improve model reliability and build a backend API and web application for RUL inference.
