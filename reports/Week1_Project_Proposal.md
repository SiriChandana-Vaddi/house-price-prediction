# Project Proposal: House Price Prediction using Linear Regression

**Student:** Siri Chandana Vaddi
**Program:** BTech CSE
**Date:** June 2026

---

## 1. Problem Statement
Predict the selling price of a residential house based on features such as average area income, house age, number of rooms, number of bedrooms, and area population. This is a supervised regression problem.

---

## 2. Objectives
- Build a Linear Regression model to predict house prices
- Perform thorough EDA and feature engineering
- Expand to advanced models (Random Forest, XGBoost)
- Deploy a working web application using Streamlit

---

## 3. Datasets
- **Primary:** USA Housing Dataset (7 columns, ~5000 rows) — simple and clean, ideal for learning linear regression
- **Advanced:** Ames Housing Dataset (Kaggle) — 79 features, real-world complexity

---

## 4. Tools & Technologies
- Python 3.x, Jupyter Notebook / Google Colab
- Libraries: pandas, NumPy, matplotlib, seaborn, scikit-learn
- Deployment: Streamlit
- Version Control: GitHub

---

## 5. Literature Review Summary

**Reference 1:** "An Introduction to Statistical Learning" (James et al.)
- Covers linear regression theory, OLS estimation, R² interpretation, and assumptions (linearity, homoscedasticity, independence, normality of residuals).

**Reference 2:** Kaggle "House Prices: Advanced Regression Techniques" notebooks
- Practical guidance on handling missing values, feature engineering, and applying ensemble models on the Ames Housing dataset.

**Reference 3:** scikit-learn Documentation — LinearRegression
- Explains the `fit()`, `predict()` API, and evaluation using MAE, MSE, RMSE metrics.

---

## 6. Linear Regression Key Concepts
- **OLS (Ordinary Least Squares):** Minimizes the sum of squared residuals between predicted and actual values.
- **Assumptions:** Linearity, independence of errors, homoscedasticity (constant variance), normality of residuals.
- **Evaluation Metrics:** MAE (mean absolute error), MSE, RMSE, R² (coefficient of determination).

---

## 7. Expected Outcome
- R² ≥ 0.7 on the USA Housing dataset with linear regression
- A deployed Streamlit app that predicts price given user inputs
- Full documentation and GitHub repo ready for evaluation

---

## 8. Timeline
8 weeks as per the project plan (Weeks 1–8), ~15–20 hours per week.
