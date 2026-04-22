# 🚀 Day 2 - Data Preprocessing in Machine Learning

## 📌 Overview

This project focuses on **Data Preprocessing**, one of the most critical steps in Machine Learning.
Raw data is often incomplete, inconsistent, and not suitable for model training. This script transforms raw data into a **clean and usable format**.

---

## 🎯 Objectives

* Handle **missing values**
* Convert **categorical data into numerical form**
* Perform **feature scaling**
* Prepare dataset for Machine Learning models

---

## 🧠 Why Data Preprocessing is Important?

* Machine Learning models **cannot handle missing or categorical data directly**
* Features with different scales can **bias the model**
* Clean data improves **accuracy and performance**

---

## ⚙️ Concepts Covered

---

### 🔹 1. Handling Missing Values

Real-world datasets often contain missing values.

#### 📌 Detection:

```python
df.isnull().sum()
```

#### 📌 Handling:

We replace missing values with the **mean** of the column:

```python
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Salary'].fillna(df['Salary'].mean(), inplace=True)
```

---

### 🔹 2. Encoding Categorical Data

Machine Learning models only understand **numerical data**.

---

#### 🔸 Label Encoding

Used for **target variables**:

```python
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df['Purchased'] = label_encoder.fit_transform(df['Purchased'])
```

👉 Example:

* Yes → 1
* No → 0

---

#### 🔸 One-Hot Encoding

Used for **input features (categorical columns)**:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), ['City'])],
    remainder='passthrough'
)
```

👉 Converts:

* Pune → [1,0,0]
* Mumbai → [0,1,0]
* Delhi → [0,0,1]

---

### 🔹 3. Feature Scaling

Features like Salary and Age have different ranges.

👉 Without scaling:

* Salary dominates the model

---

#### 📌 Standardization (Z-score scaling)

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

👉 Formula:

[
Z = \frac{X - \mu}{\sigma}
]

---

### 🔹 4. Train-Test Split

We split data to evaluate model performance:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

## 🔄 Complete Workflow

```text
Raw Data
   ↓
Handle Missing Values
   ↓
Encode Categorical Data
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Ready for Model Training
```

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
python preprocessing.py
```

---

## 📊 Output

* Clean dataset
* Encoded categorical variables
* Scaled features
* Ready-to-use ML dataset

---

## ⚠️ Important Interview Points

* **Why Encoding?**
  ML models cannot process text directly.

* **Why Scaling?**
  Distance-based algorithms (KNN, SVM) require normalized data.

* **Why Split Data?**
  To test model on unseen data.

---

## 🎯 Conclusion

This project demonstrates how to:

* Clean raw data
* Convert it into numerical format
* Normalize features for better model performance

👉 Data Preprocessing is the **foundation of every successful ML model**.

---

## 🔥 Next Step

Move to **Day 3: Linear Regression (Model Building & Evaluation)** 🚀

---
