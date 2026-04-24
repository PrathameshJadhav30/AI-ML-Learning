# 🚀 Day 3 - Linear Regression (Machine Learning)

## 📌 Overview

This project demonstrates **Linear Regression**, one of the most fundamental algorithms in Machine Learning used for **predicting continuous values**.

We implement a complete workflow:

* Data loading
* Data visualization
* Model training
* Prediction
* Evaluation
* Residual analysis

---

## 🎯 Objectives

* Understand **Linear Regression conceptually**
* Implement regression model using `scikit-learn`
* Evaluate model performance using metrics
* Interpret model results

---

## 🧠 What is Linear Regression?

Linear Regression is a **supervised learning algorithm** used to predict a **continuous target variable** based on input features.

It finds the **best-fit line** that minimizes the error between predicted and actual values.

---

## 📐 Mathematical Representation

y = m_1x_1 + m_2x_2 + \cdots + m_nx_n + c

Where:

* (y) = Target variable
* (x) = Input features
* (m) = Coefficients (weights)
* (c) = Intercept

---

## 📊 Dataset Used

We use the **California Housing Dataset** from `sklearn`.

* Features: Income, House Age, Rooms, Population, etc.
* Target: House Price

---

## ⚙️ Workflow Explained

---

### 🔹 1. Data Loading

Dataset is loaded using:

```python
from sklearn.datasets import fetch_california_housing
```

---

### 🔹 2. Data Visualization

Used to understand relationships between variables:

* Correlation Heatmap
* Scatter Plot (Feature vs Target)

👉 Helps identify important features

---

### 🔹 3. Feature & Target Selection

```python
X = df.drop('PRICE', axis=1)
y = df['PRICE']
```

---

### 🔹 4. Train-Test Split

```python
from sklearn.model_selection import train_test_split
```

* 80% Training Data
* 20% Testing Data

👉 Ensures model generalization

---

### 🔹 5. Model Training

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

👉 Model learns relationship between features and target

---

### 🔹 6. Prediction

```python
y_pred = model.predict(X_test)
```

👉 Predicts values on unseen data

---

### 🔹 7. Model Evaluation

#### 📌 Mean Squared Error (MSE)

Measures average squared error:

MSE = \frac{1}{n} \sum_{i=1}^{n}(y_i - \hat{y}_i)^2

* Lower value = better model

---

#### 📌 R² Score (Coefficient of Determination)

Measures how well model explains variance:

R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}

* Range: 0 to 1
* Higher value = better fit

---

### 🔹 8. Residual Analysis

Residual = Actual − Predicted

* Used to check model errors
* Should be randomly distributed

---

### 🔹 9. Feature Importance

```python
model.coef_
```

👉 Coefficients indicate:

* Positive → Direct relationship
* Negative → Inverse relationship

---

## 📈 Visualizations Used

* 📊 Correlation Heatmap
* 📉 Scatter Plot
* 📈 Regression Line
* 📊 Residual Distribution
* 📊 Actual vs Predicted Plot

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

---

## 🚀 How to Run

1. Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

2. Run script:

```bash
python day3_linear_regression.py
```

---

## 📊 Output

* Predicted values
* Evaluation metrics (MSE, R²)
* Graphical insights

---

## ⚠️ Important Interview Questions

### ❓ What is Linear Regression?

Predicts continuous values using a linear relationship.

---

### ❓ What is the role of coefficients?

They represent the **impact of each feature** on the target.

---

### ❓ What is overfitting?

Model performs well on training but poorly on testing data.

---

### ❓ What is R² score?

Measures how well the model explains variance.

---

### ❓ What are residuals?

Difference between actual and predicted values.

---

## 🎯 Conclusion

This project demonstrates:

* Building a regression model from scratch
* Evaluating performance using metrics
* Interpreting model behavior

👉 Linear Regression is the **foundation of many advanced ML algorithms**

---

## 🔥 Next Step

Move to **Day 4: Logistic Regression (Classification Problems)** 🚀

---
