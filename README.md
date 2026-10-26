# 🏠 House Price Prediction Using Linear Regression

## 📌 Project Overview

This project focuses on predicting house prices using **Linear Regression**, a machine learning algorithm.

The project uses data analysis and a regression model to understand the relationship between house features and their prices.

## 🎯 Objectives

- Analyze house price data.
- Perform Exploratory Data Analysis (EDA).
- Build a Linear Regression model.
- Evaluate the model's prediction performance.

## 🛠️ Technologies Used

- Python
- Jupyter Notebook
- Machine Learning
- Linear Regression

## ⚙️ Project Workflow

1. Load and explore the dataset.
2. Perform data analysis and visualization.
3. Identify the target variable (`SalePrice`).
4. Prepare the data for model training.
5. Train the Linear Regression model.
6. Evaluate the model's performance.

## 📊 Model Evaluation

The model's performance can be evaluated using:

- **MAE:** Mean Absolute Error
- **MSE:** Mean Squared Error
- **RMSE:** Root Mean Squared Error
- **R² Score:** Coefficient of Determination

## 📊 Model Evaluation Results

The models were evaluated using a held-out test dataset containing 972 samples.

### Best Performing Model

**Polynomial Features (Degree 2) + Ridge Regression**

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 0.0712 |
| Root Mean Squared Error (RMSE) | 0.0933 |
| R² Score | 0.8982 |
| Adjusted R² Score | 0.8973 |

### Model Comparison

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Polynomial (Degree 2) + Ridge | 0.0712 | 0.0933 | 0.8982 |
| Linear Regression | 0.0773 | 0.1000 | 0.8832 |
| Ridge (Tuned) | 0.0773 | 0.1000 | 0.8832 |
| Lasso (Tuned) | 0.0773 | 0.1001 | 0.8830 |
| Baseline (Mean Predictor) | 0.2259 | 0.2929 | -0.0024 |

### Key Findings

- Polynomial features of degree 2 with Ridge regression achieved the highest test R² score among the evaluated models.
- The best model achieved an RMSE of 0.0933.
- Five-fold cross-validation was used to compare models and tune hyperparameters.
- Residual analysis, influential-point analysis, and learning curves were used to examine model behavior.

*Note: The reported metric values are in the target variable's scale used by the notebook. Confirm whether the target was transformed before interpreting the errors in currency units.*

## 🚀 How to Run

1. Clone or download this repository.
2. Install Python and Jupyter Notebook if they are not already installed.
3. Open the project notebook in Jupyter Notebook.
4. Ensure the required dataset is available at the path specified in the notebook.
5. Run the notebook cells in order.

## 📚 What I Learned

- Basics of machine learning with Linear Regression.
- Exploratory Data Analysis.
- Understanding regression evaluation metrics.
- Working with datasets in Jupyter Notebook.

## 👩‍💻 Author

**Siri Chandana**

- GitHub: [SiriChandana-Vaddi](https://github.com/SiriChandana-Vaddi)
- Portfolio: [Visit My Portfolio](https://sirichandana-vaddi.github.io/siri-portfolio/)
