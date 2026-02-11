# Design and Application of a Machine Learning System (CE802)

This repository contains the implementation and comparative evaluation of multiple machine learning algorithms developed as part of the CE802 Machine Learning & Data Mining module at the University of Essex.

The objective of this project was to design, evaluate, and compare various machine learning models to determine the most suitable approach for a practical classification and regression problem.

---

## 📌 Project Overview

The project involves:

- Exploratory Data Analysis (EDA)
- Handling missing values
- Feature engineering and skewness correction
- Feature selection using correlation analysis
- Model training and evaluation
- Comparative performance analysis across multiple ML algorithms

---

## 📊 Dataset Description

### Primary Dataset:
- 22 Features
- 1000 Instances
- Missing values identified in feature F21
- Skewed distributions handled using transformation techniques

### Additional Dataset:
- 37 Features
- 1500 Instances
- Categorical variables converted to numerical form
- Feature importance evaluated using baseline model

---

## 🔍 Exploratory Data Analysis

- Identification of missing values (F21 had 500 null values)
- Visualization of null values
- Boxplot-based percentile estimation
- Correlation heatmap analysis
- Skewness detection (positive & negative skew)
- Feature transformation:
  - Logarithm
  - Square-root
  - Cube-root
  - Reciprocal
  - Square & cube (for negative skew)

---

## ⚙️ Feature Engineering

- Pearson correlation analysis
- Removal of highly correlated features
- Standardization (Z-score scaling)
- Normalization (Min-Max scaling)
- Robust Scaler (IQR-based scaling)
- Baseline feature importance using XGBRegressor

---

## 🔀 Data Splitting Strategy

Train-Test Split:
- 80% Training
- 20% Testing

---

## 🤖 Machine Learning Models Evaluated

Classification Models:
- Random Forest
- AdaBoost
- Gradient Boosting
- Extra Trees
- Decision Tree
- Logistic Regression
- SVC
- K-Nearest Neighbors (kNN)
- Naïve Bayes
- Ridge Classifier
- Perceptron
- Passive Aggressive Classifier
- And others (Sklearn classification models)

Regression Models:
- GradientBoostingRegressor
- ExtraTreesRegressor
- RandomForestRegressor
- BaggingRegressor
- LinearRegression
- Lasso
- BayesianRidge
- HuberRegressor
- AdaBoostRegressor
- ElasticNet
- DecisionTreeRegressor

---

## 📈 Model Evaluation Metrics

For Classification:
- Accuracy
- Precision
- Recall
- AUC
- ROC Curve Comparison

For Regression:
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

---

## 🧠 Baseline Model

A baseline model was created using:

- **XGBRegressor**

Feature importance analysis was conducted to identify the most impactful features for prediction.

---

## 📊 Comparative Analysis

- Performance comparison across multiple models
- Visualization of accuracy, precision, recall, and AUC
- ROC curve comparison
- Identification of best-performing model based on evaluation metrics

---

## 📁 Repository Structure

```
notebooks/        → Jupyter notebooks for experiments
data/             → Dataset files (if included)
models/           → Saved trained models (if included)
report/           → Final project report
```

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
jupyter notebook
```

---

## 🎓 Academic Context

Developed as part of:

CE802 – Machine Learning & Data Mining  
School of Computer Science & Electronic Engineering  
University of Essex
