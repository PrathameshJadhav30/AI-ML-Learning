# 🚀 Day 6 - Decision Tree & Random Forest (Machine Learning)

## 📌 Overview

This project demonstrates two powerful **tree-based Machine Learning algorithms**:

* 🌳 **Decision Tree**
* 🌲 **Random Forest**

These algorithms are widely used for **classification problems** and are very important for interviews and real-world applications.

---

## 🎯 Objectives

* Understand how **Decision Trees** work
* Learn **Random Forest (Ensemble Learning)**
* Compare model performance
* Analyze **feature importance**
* Understand **overfitting**

---

## 🧠 What is a Decision Tree?

A Decision Tree is a **supervised learning algorithm** that splits data into branches to make decisions.

👉 It works like a flowchart:

```text id="b9x1k2"
Is feature <= value?
   ├── Yes → Go Left
   └── No  → Go Right
```

---

## 🌳 Key Concepts of Decision Tree

* Splits data based on conditions
* Uses metrics like:

  * **Gini Index**
  * **Entropy**
* Easy to understand and visualize

---

### 🔹 Gini Index (Impurity Measure)

Gini = 1 - \sum p_i^2

👉 Lower Gini = better split

---

## ⚠️ Problem with Decision Tree

👉 **Overfitting**

* Model learns training data too well
* Performs poorly on new data

---

## 🌲 What is Random Forest?

Random Forest is an **ensemble learning algorithm** that combines multiple decision trees.

👉 Final prediction = **majority voting**

---

## 🌲 Why Random Forest?

* Reduces overfitting
* Improves accuracy
* More stable than single tree

---

## 📊 Dataset Used

We use the **Wine Dataset** from `sklearn`.

* Features: Chemical properties of wine
* Target: Wine category (0, 1, 2)

---

## ⚙️ Workflow Explained

---

### 🔹 1. Data Loading

```python id="p2x4q7"
from sklearn.datasets import load_wine
```

---

### 🔹 2. Feature & Target Selection

```python id="k8m3z1"
X = df.drop('TARGET', axis=1)
y = df['TARGET']
```

---

### 🔹 3. Train-Test Split

```python id="w6n2y9"
from sklearn.model_selection import train_test_split
```

---

### 🔹 4. Decision Tree Model

```python id="c9v5d2"
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
```

---

### 🔹 5. Random Forest Model

```python id="u7l1x8"
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
```

---

### 🔹 6. Model Evaluation

We compare models using:

* Accuracy Score

```python id="z3k8h5"
from sklearn.metrics import accuracy_score
```

---

## 📊 Feature Importance

Random Forest provides feature importance:

```python id="n4r2p6"
rf.feature_importances_
```

👉 Shows which features influence prediction most

---

## ⚠️ Overfitting Check

```python id="d1j7t3"
train_accuracy = dt.score(X_train, y_train)
test_accuracy = dt.score(X_test, y_test)
```

👉 If:

* Training Accuracy >> Testing Accuracy
  → Model is overfitting

---

## 📈 Visualizations (Optional)

* Tree diagram
* Feature importance bar chart
* Accuracy comparison

---

## 🛠️ Technologies Used

* Python
* Pandas
* Scikit-learn

---

## 🚀 How to Run

1. Install dependencies:

```bash id="f6y9k1"
pip install pandas scikit-learn
```

2. Run script:

```bash id="h2x8q4"
python day6_simple.py
```

---

## 📊 Output

* Decision Tree accuracy
* Random Forest accuracy
* Feature importance
* Overfitting analysis

---

## ⚠️ Important Interview Questions

### ❓ What is Decision Tree?

A model that splits data into branches based on conditions.

---

### ❓ What is Random Forest?

A collection of multiple decision trees combined.

---

### ❓ Why Random Forest is better?

* Reduces overfitting
* Improves accuracy

---

### ❓ What is Gini Index?

Measure of impurity used for splitting nodes.

---

### ❓ What is overfitting?

Model performs well on training but poorly on test data.

---

## 🎯 Conclusion

This project demonstrates:

* Tree-based learning algorithms
* Model comparison
* Importance of ensemble methods

👉 Decision Tree + Random Forest are **core ML concepts used in industry**

---

## 🔥 Next Step

Move to **Day 7: K-Means Clustering (Unsupervised Learning)** 🚀

---
