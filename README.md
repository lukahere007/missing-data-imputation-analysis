# 📊 Missing Data Imputation Analysis

This project demonstrates the impact of different missing data mechanisms—MCAR, MAR, and MNAR—on linear regression estimates using simulated data. It compares:

- Complete case analysis
- Mean imputation
- Regression-based imputation

The notebook explores how each method influences the estimated coefficients (`x1`, `x2`) and provides visualizations for deeper insight.

## 📁 File

📎 **Explore the notebook:**  
[Vaccine Imputation Analysis Notebook](https://github.com/lukahere007/missing-data-imputation-analysis/blob/main/missing_data_imputation_analysis.ipynb)

## 📌 Key Features

- Simulates datasets under MCAR, MAR, and MNAR assumptions
- Applies mean and regression imputation strategies
- Compares regression outputs from complete, missing, and imputed data
- Visualizes differences in coefficient estimates

## 📈 Visualizations

- Boxplots of imputations
- Line plots for regression fits
- Bar plots comparing coefficient estimates for `x1` and `x2`

---

### 🧠 Summary

Understanding how missing data affects model estimates is essential for robust statistical analysis. This notebook highlights how each imputation strategy performs under different missingness assumptions.

---

### 🔧 Requirements

Run using:

```bash
jupyter lab

