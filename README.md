# 📈 Polynomial Regression vs Linear Regression — Employee Salary Prediction

A Machine Learning project that compares **Linear Regression** and **Polynomial Regression** to predict employee salaries based on their position level.

---

## 🔍 Problem Statement

Given an employee's position level (1–10), predict their expected salary.
A simple linear model fails to capture the non-linear salary growth pattern, so **Polynomial Regression** is used to improve accuracy.

---

## 🧠 Models Used

### 1. Linear Regression (degree = 1)
Fits a straight line through the data. Struggles when the relationship is curved.

### 2. Polynomial Regression (degree = 2)
Expands features as `[1, x, x²]` and fits a curve — capturing non-linear patterns far better.

---

## 🔁 How It Works

Input: Position Level (X)
         │
         ├──► Linear Regression     → Straight line fit  (degree 1)
         │
         └──► PolynomialFeatures()  → Transforms X to [1, x, x²]
                      │
                      └──► Linear Regression on X_poly → Curved fit

---

## 📉 Visualizations

 Linear Regression = Straight line — clearly underfits the salary curve.

 Polynomial Regression = Curved fit — follows the non-linear salary growth pattern closely.

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

## 📌 Why Not Linear Regression?

Linear regression assumes salary increases at a **constant rate** per level — unrealistic for senior/C-suite roles that have **exponentially higher** pay. Polynomial regression captures this curve.

---

## 👩‍💻 Author

Kalyani Zade |
Machine Learning enthusiast | Python Developer
