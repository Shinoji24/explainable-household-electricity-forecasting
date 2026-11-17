# Explainable Household Electricity Consumption Forecasting and Anomaly Detection

**WattScope** is a machine learning application designed to forecast household electricity consumption, detect unusual usage patterns, and explain forecasting predictions.

## Project Overview

This project uses historical household electricity consumption data to understand usage patterns and predict future electricity demand.

It combines machine learning, anomaly detection, and explainable AI techniques to provide insights into household electricity consumption.

The application is developed using Streamlit and deployed on AWS EC2.

## Objectives

- Analyse historical household electricity consumption data.
- Forecast electricity consumption for the next 24 hours.
- Identify unusual electricity consumption patterns.
- Compare machine learning models to select a suitable forecasting model.
- Explain model predictions using SHAP.

## Dataset

**Source:** [UCI Machine Learning Repository – Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)

The dataset contains approximately 2 million household electricity measurements recorded at one-minute intervals.

**Target variable:** `Global_active_power`

### Data Preprocessing

- Handled missing values and investigated missing time intervals.
- Combined Date and Time into a datetime index.
- Examined outliers using the Interquartile Range (IQR) method.
- Interpolated short missing intervals.
- Converted minute-level readings into hourly average consumption.
- Created lag features for forecasting.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- SHAP
- Streamlit
- Pickle
- AWS EC2

## Machine Learning Models

### 1. Electricity Consumption Forecasting

The following approaches were evaluated:

- Lag 1 Baseline
- Random Forest Regressor
- XGBoost Regressor

**Selected model:** XGBoost Regressor

Lag features are used to capture previous electricity consumption values, including the previous hour and the same hour from the previous day.

### 2. Model Evaluation

The forecasting model was evaluated using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

| Metric | XGBoost Result |
|---|---:|
| MAE | 0.3676 kW |
| RMSE | 0.5184 kW |

These metrics measure prediction error; lower values indicate better predictions.

### 3. Anomaly Detection

**Algorithm:** Isolation Forest

Isolation Forest is used to identify unusual electricity consumption observations in hourly data.

The model uses an assumed contamination setting of 0.02 to identify potentially unusual observations. Detected anomalies indicate unusual usage patterns and do not necessarily mean an appliance is faulty.

### 4. Explainable AI

**Technique:** SHAP (SHapley Additive exPlanations)

SHAP is used to explain individual XGBoost forecasting predictions by showing how different lag features contribute to the predicted electricity consumption.

## Application Features

The WattScope application includes four main sections:

### 1. Overview
Provides an introduction to the application and its purpose.

### 2. Consumption Forecasting
- Upload electricity consumption data through a CSV file.
- Enter previous consumption values manually.
- Generate demo predictions using sample input values.
- View historical actual versus predicted consumption.
- Generate a recursive forecast for the next 24 hours.

### 3. Anomaly Detection
- Identify unusual electricity consumption patterns.
- Display normal and anomalous observations.
- Visualize detected anomalies.

### 4. Explainable AI
- Select a historical forecasting observation.
- View a SHAP waterfall plot.
- Understand how lag features influence an individual prediction.

## Application Workflow

1. Load historical household electricity consumption data.
2. Perform data cleaning and preprocessing.
3. Resample minute-level data into hourly readings.
4. Create lag-based forecasting features.
5. Train and evaluate forecasting models.
6. Select the XGBoost model for forecasting.
7. Apply Isolation Forest for anomaly detection.
8. Use SHAP to explain XGBoost predictions.
9. Integrate the saved models into a Streamlit application.

## Deployment

The application is developed using Streamlit and deployed on an AWS EC2 instance.

The trained forecasting and anomaly detection models are saved as Pickle files and loaded by the application for prediction.

## Running the Application Locally

Clone the repository:

```bash
git clone https://github.com/Shinoji24/explainable-household-electricity-forecasting.git
```

Navigate to the project directory:

```bash
cd explainable-household-electricity-forecasting
```

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap streamlit
```

Ensure that the application file and required saved model files (`best_model.pkl` and `isolation_forest.pkl`) are available in the project directory.

Run the application:

```bash
streamlit run app.py
```

## Project Status

The WattScope application has been implemented with electricity consumption forecasting, anomaly detection, and SHAP-based explainability.

The application has also been deployed on AWS EC2.

Further improvements and enhancements can be added in future development.
