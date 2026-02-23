# Heart Disease ML Kaggle Project

This repository contains the code and resources for the Heart Disease ML Kaggle competition.

## Project Structure

- **data/**: Contains the datasets used for training and testing.
- **notebooks/**:
    - `EDA.ipynb`: Exploratory Data Analysis of the heart disease dataset.
    - `main (3).ipynb` and `models.ipynb`: Experimental notebooks for model development and testing.
- **exp/**: 
    - Contains further experiment logs, roadmaps, and kaggle submission files (`submission.csv`, etc.).
    - Includes `main.ipynb` for core experimentation.
- **src/**: 
    - Core source code for the project.
    - Includes data loading, preprocessing, feature engineering, and training scripts.
    - **MLflow & DagsHub**: MLflow code is integrated with DagsHub for experiment tracking.
- **heart_app.py**: A Streamlit application to interact with the trained models.
- **mlruns/**: Automated MLflow tracking logs.
- **models/**: Directory where trained models are saved.

## DagsHub Integration

All experiments and model parameters are tracked using MLflow and hosted on DagsHub.

**DagsHub Repository**: [https://dagshub.com/Bhavik2209/Heart_disease_ML_kaggle](https://dagshub.com/Bhavik2209/Heart_disease_ML_kaggle)

## How to Run the App

To run the Streamlit app, use the following command:

```bash
streamlit run heart_app.py
```
