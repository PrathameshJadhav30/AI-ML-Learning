# 🚀 Day 5 - K-Nearest Neighbors (KNN) & Naive Bayes

## 📌 Overview

This project demonstrates two important Machine Learning classification algorithms:

* 🔹 **K-Nearest Neighbors (KNN)** – Distance-based learning
* 🔹 **Naive Bayes** – Probability-based learning

Both algorithms are widely used for classification problems and help understand **different approaches to decision-making in ML**.

---

## 🎯 Objectives

* Understand **distance-based vs probability-based models**
* Implement **KNN and Naive Bayes using scikit-learn**
* Evaluate models using classification metrics
* Compare performance of both algorithms

---

## 🧠 What is K-Nearest Neighbors (KNN)?

KNN is a **supervised learning algorithm** used for classification and regression.

👉 It works by:

* Finding the **K nearest data points**
* Assigning the class based on **majority voting**

---

## 📐 KNN Working Concept

1. Choose value of **K**
2. Calculate distance between data points
3. Select K nearest neighbors
4. Assign class based on majority vote

---

### 🔹 Distance Formula (Euclidean Distance)

d = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}

---

## ⚠️ Important Points for KNN

* Requires **feature scaling**
* Sensitive to **value of K**
* Slow for large datasets

---

## 🧠 What is Naive Bayes?

Naive Bayes is a **probability-based classification algorithm** based on **Bayes' Theorem**.

👉 It assumes:

* All features are **independent** (naive assumption)

---

### 🔹 Bayes Theorem

genui{"math_block_widget_always_prefetch_v2":{"content":"P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}"}}

Where:

* (P(A|B)) = Posterior probability
* (P(B|A)) = Likelihood
* (P(A)) = Prior probability
* (P(B)) = Evidence

---

## 📊 Dataset Used

We use the **Iris Dataset** from `sklearn`.

* Features:

  * Sepal length
  * Sepal width
  * Petal length
  * Petal width

* Target:

  * 0 → Setosa
  * 1 → Versicolor
  * 2 → Virginica

---

## ⚙️ Workflow Explained

---

### 🔹 1. Data Loading

```python id="a2x9h7"
from sklearn.datasets import load_iris
```

---

### 🔹 2. Data Visualization

* Pairplot used to understand feature relationships
* Helps visualize class separation

---

### 🔹 3. Feature & Target Selection

```python id="g7k2d9"
X = df.drop('TARGET', axis=1)
y = df['TARGET']
```

---

### 🔹 4. Train-Test Split

```python id="p9w4x1"
from sklearn.model_selection import train_test_split
```

---

### 🔹 5. Feature Scaling

```python id="r8j3m6"
from sklearn.preprocessing import StandardScaler
```

👉 Important for KNN because it uses distance calculations

---

### 🔹 6. KNN Model Implementation

```python id="u5l2z8"
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
```

---

### 🔹 7. Naive Bayes Model Implementation

```python id="y3t9q4"
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(X_train, y_train)
```

---

## 📊 Evaluation Metrics

* Accuracy
* Confusion Matrix
* Classification Report

---

### 🔹 Accuracy

$$
Accuracy = \frac{Correct Predictions}{Total Predictions}
$$

---

### 🔹 Confusion Matrix

|                | Predicted Class A | Predicted Class B |
| -------------- | ----------------- | ----------------- |
| Actual Class A | Correct           | Incorrect         |
| Actual Class B | Incorrect         | Correct           |

---

### 🔹 Classification Report

Includes:

* Precision
* Recall
* F1 Score

---

## 📈 Model Comparison

* Both models are trained and evaluated
* Accuracy is compared using bar chart
* Helps identify better performing model

---

## 🔧 Hyperparameter Tuning (KNN)

We test different values of **K**:

* Small K → Overfitting
* Large K → Underfitting

👉 Optimal K gives best performance

---

## 📊 Visualizations Used

* 📊 Pairplot
* 📊 Confusion Matrix
* 📊 Accuracy Comparison Bar Chart
* 📊 Error Rate vs K Graph

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

```bash id="z1k7x2"
pip install numpy pandas matplotlib seaborn scikit-learn
```

2. Run script:

```bash id="m4c8p9"
python day5_knn_naive_bayes.py
```

---

## 📊 Output

* Predictions from KNN and Naive Bayes
* Accuracy comparison
* Confusion matrices
* Performance visualization

---

## ⚠️ Important Interview Questions

### ❓ What is K in KNN?

Number of nearest neighbors used for classification.

---

### ❓ Why scaling is required in KNN?

Because KNN uses distance → features must be normalized.

---

### ❓ What is Naive Bayes assumption?

All features are independent.

---

### ❓ Difference between KNN and Naive Bayes?

| KNN                | Naive Bayes       |
| ------------------ | ----------------- |
| Distance-based     | Probability-based |
| Lazy learner       | Eager learner     |
| Slow on large data | Fast              |

---

### ❓ When to use Naive Bayes?

* Text classification
* Spam detection

---

## 🎯 Conclusion

This project demonstrates:

* Two different ML approaches:

  * Distance-based (KNN)
  * Probability-based (Naive Bayes)
* Model comparison and evaluation
* Importance of preprocessing and tuning

👉 These are **core algorithms for classification problems**

---

## 🔥 Next Step

Move to **Day 6: Decision Tree & Random Forest (Tree-Based Models)** 🚀

---
