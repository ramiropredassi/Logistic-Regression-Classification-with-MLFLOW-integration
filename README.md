# Logistic Regression Classification with MLFLOW integration

**Author:** Ramiro Predassi  
**Date:** June 2025  

## Description

This project demonstrates the full pipeline of a classification task using Logistic Regression, along with comprehensive integration of MLFLOW for model tracking and experiment management. The dataset used is from the [Spaceship Titanic competition](https://www.kaggle.com/competitions/spaceship-titanic/data) on Kaggle.

The workflow includes:
- Data analysis and visualization
- Preprocessing and transformation
- Hyperparameter tuning
- Model evaluation
- MLFLOW tracking for reproducibility

## Table of Contents

1. [Data Sources](#data-sources)  
2. [Exploratory Data Analysis](#exploratory-data-analysis)  
   - Missing Values  
   - Numerical Variables  
   - Feature Distributions  
   - Categorical Variables  
   - Cardinality  
   - Outlier Detection  
   - Feature Relationships  
3. [Preprocessing](#preprocessing)  
   - Train/Test Transformations  
4. [Logistic Regression Classification](#logistic-regression-classification)  
   - Hyperparameter Tuning  
   - Training and Evaluation  
5. [MLFLOW Tracking](#mlflow-tracking)  
   - Model Logging and Loading

## Data Sources

The dataset is sourced from the [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic/data) competition. It includes features such as demographics, passenger IDs, services used, and survival outcome (`Transported`).

## Exploratory Data Analysis

Performed using:
- `pandas`, `seaborn`, `matplotlib` for visualizations
- Missing value analysis
- Statistical summaries and distributions
- Outlier detection using Box Plot

## Preprocessing

- Applied transformations using `PowerTransformer` 
- Encoded categorical variables
- Addressed missing data
- Ensured consistency between train and test datasets

## Logistic Regression Classification

Used `sklearn.LogisticRegression` with:
- `GridSearchCV` for hyperparameter tuning
- Evaluation via accuracy, precision, recall, and classification report

## MLFLOW Tracking

- Tracking URI set to `http://127.0.0.1:5000`
- Parameters, metrics, models, and artifacts logged
- Signature inferred using `mlflow.models.infer_signature`
- Model saved and reloaded using `mlflow.sklearn`
