# 🚀 Day 1 - Machine Learning Fundamentals & First Model

## 📌 Overview

This project demonstrates the **basic Machine Learning workflow** using a real dataset.
It covers everything from **data loading → preprocessing → model training → evaluation**.

---

## 🎯 Objectives

* Understand the **Machine Learning pipeline**
* Perform **basic data analysis**
* Split dataset into **training and testing sets**
* Train a **Linear Regression model**
* Evaluate model performance

---

## 🧠 Machine Learning Workflow

```
Data Collection → Data Preprocessing → Train-Test Split → Model Training → Prediction → Evaluation
```

---

## 📊 Dataset Used

We use the **California Housing Dataset** from `sklearn`.

* Features: Income, House Age, Rooms, Population, etc.
* Target: House Price

---

## ⚙️ Step-by-Step Explanation

---

### 🔹 1. Import Libraries

We import required libraries for:

* Data handling → `pandas`, `numpy`
* Visualization → `matplotlib`, `seaborn`
* Machine Learning → `sklearn`

---

### 🔹 2. Load Dataset

Dataset is loaded using:

```python
from sklearn.datasets import fetch_california_housing
```

We convert it into a **Pandas DataFrame** for easier manipulation.

---

### 🔹 3. Data Understanding

We analyze:

* Shape of dataset
* Data types
* Statistical summary

Functions used:

```python
df.shape
df.info()
df.describe()
```

---

### 🔹 4. Handling Missing Values

Check missing values:

```python
df.isnull().sum()
```

If present:

```python
df.fillna(df.mean(), inplace=True)
```

---

### 🔹 5. Data Visualization

* Correlation Heatmap → Understand feature relationships
* Distribution Plot → Understand target variable

---

### 🔹 6. Feature & Target Selection

```python
X = df.drop('PRICE', axis=1)
y = df['PRICE']
```

---

### 🔹 7. Train-Test Split

We split data into:

* Training Data → 80%
* Testing Data → 20%

```python
from sklearn.model_selection import train_test_split
```

Why?

* To evaluate model on unseen data

---

### 🔹 8. Model Training

We use **Linear Regression**:

```python
from sklearn.linear_model import LinearRegression
```

Model learns relationship:

```
y = mx + c
```

---

### 🔹 9. Prediction

Model predicts output using:

```python
model.predict(X_test)
```

---

### 🔹 10. Model Evaluation

We evaluate using:

* **Mean Squared Error (MSE)** → Error measurement
* **R² Score** → Accuracy of model

```python
from sklearn.metrics import mean_squared_error, r2_score
```

---

### 🔹 11. Visualization of Results

We compare:

* Actual values vs Predicted values

---

## 📈 Key Concepts Learned

* Machine Learning Pipeline
* Supervised Learning (Regression)
* Train-Test Split
* Linear Regression
* Model Evaluation Metrics

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

2. Run the Python script:

```bash
python main.py
```

---

## 📌 Output

* Model predictions
* Evaluation metrics (MSE, R²)
* Graphs for visualization

---

## 🎯 Conclusion

This project builds a **strong foundation in Machine Learning** by implementing:

* Data preprocessing
* Model training
* Performance evaluation

It is the **first step toward building real-world ML projects**.

---

## 🔥 Next Step

Move to **Day 2: Data Preprocessing (Encoding, Scaling, Cleaning)**

---
