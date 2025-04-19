# Missing Data Imputation Analysis

This project explores how different imputation strategies affect statistical modeling when data is Missing Completely at Random (MCAR) or Missing at Random (MAR). Using synthetic data, we analyze how mean and regression imputation impact coefficient estimates for linear models.

---

## 📁 Contents

- [`missing_data_imputation_analysis.ipynb`](https://github.com/lukahere007/missing-data-imputation-analysis/blob/main/missing_data_imputation_analysis.ipynb) – Jupyter notebook with full simulation, modeling, and plots
- `README.md` – Overview and instructions

---

## 📊 Objective

The goal is to demonstrate:
- How missing data mechanisms (MCAR, MAR) can distort estimates
- Comparison between:
  - ✅ Full data (no missingness)
  - 🚫 Mean imputation (for MCAR)
  - 📈 Regression imputation (for MAR)

---

## 📦 Dependencies

Install the following packages before running:

```bash
pip install numpy pandas matplotlib seaborn statsmodels
