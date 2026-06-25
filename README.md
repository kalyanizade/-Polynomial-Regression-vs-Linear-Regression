# 📈 Polynomial Regression vs Linear Regression — Employee Salary Prediction

A Machine Learning project that compares **Linear Regression** and **Polynomial Regression** to predict employee salaries based on their position level.

---

## 🔍 Problem Statement

Given an employee's position level (1–10), predict their expected salary.
A simple linear model fails to capture the non-linear salary growth pattern, so **Polynomial Regression** is used to improve accuracy.

---

## 📁 Dataset

**File:** `emp_sal.csv`

| Column | Description |
|--------|-------------|
| Position | Job title (e.g., Analyst, Manager, CEO) |
| Level | Numeric level (1 to 10) |
| Salary | Annual salary in USD |

---

## 🧠 Models Used

### 1. Linear Regression (degree = 1)
Fits a straight line through the data. Struggles when the relationship is curved.

### 2. Polynomial Regression (degree = 2)
Expands features as `[1, x, x²]` and fits a curve — capturing non-linear patterns far better.

---

## 📊 Results

| Model | Predicted Salary for Level 6.5 | Accurate? |
|-------|-------------------------------|-----------|
| Linear Regression | ~$330,379 | ❌ Way too high |
| Polynomial Regression (degree=2) | ~$189,498 | ✅ Within expected range ($150K–$200K) |

---

## ⚙️ Installation

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

## 💻 Code Overview

```python
# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
lin_reg.predict([[6.5]])   # ~$330,379 ❌

# Polynomial Regression
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))  # ~$189,498 ✅
```

---

## 📌 Why Not Linear Regression?

Linear regression assumes salary increases at a **constant rate** per level — unrealistic for senior/C-suite roles that have **exponentially higher** pay. Polynomial regression captures this curve.

---

## 🗂️ Project Structure
