# 🚀 Day 4 - Logistic Regression (Classification in Machine Learning)

## 📌 Overview

This project demonstrates **Logistic Regression**, a fundamental Machine Learning algorithm used for **classification problems**.

Unlike Linear Regression, which predicts continuous values, Logistic Regression predicts **categorical outcomes** such as:

* Yes / No
* 0 / 1
* True / False

---

## 🎯 Objectives

* Understand **classification problems**
* Learn **Logistic Regression algorithm**
* Implement classification model using `scikit-learn`
* Evaluate model using multiple performance metrics
* Visualize model performance

---

## 🧠 What is Logistic Regression?

Logistic Regression is a **supervised learning algorithm** used for **binary classification**.

It predicts the **probability** that a data point belongs to a particular class.

---

## 📐 Mathematical Concept (Sigmoid Function)

Logistic Regression uses the **Sigmoid Function** to map values between 0 and 1:

\sigma(z) = \frac{1}{1 + e^{-z}}

Where:

* (z = m_1x_1 + m_2x_2 + ... + c)
* Output is a probability between 0 and 1

👉 If probability ≥ 0.5 → Class 1
👉 If probability < 0.5 → Class 0

---

## 📊 Dataset Used

We use the **Breast Cancer Dataset** from `sklearn`.

* Features: Various medical measurements
* Target:

  * 0 → Malignant (Cancerous)
  * 1 → Benign (Non-cancerous)

---

## ⚙️ Workflow Explained

---

### 🔹 1. Data Loading

```python id="e8l2jp"
from sklearn.datasets import load_breast_cancer
```

---

### 🔹 2. Data Understanding

* Dataset shape
* Feature information
* Class distribution

---

### 🔹 3. Data Visualization

* Correlation Heatmap
* Class Distribution Plot

👉 Helps understand data patterns

---

### 🔹 4. Feature & Target Selection

```python id="7h3i2m"
X = df.drop('TARGET', axis=1)
y = df['TARGET']
```

---

### 🔹 5. Train-Test Split

```python id="pjk8v1"
from sklearn.model_selection import train_test_split
```

👉 Ensures model is tested on unseen data

---

### 🔹 6. Feature Scaling

```python id="4x8zq3"
from sklearn.preprocessing import StandardScaler
```

👉 Required because Logistic Regression is distance-based

---

### 🔹 7. Model Training

```python id="7o3b9p"
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

---

### 🔹 8. Prediction

```python id="9m4xrp"
y_pred = model.predict(X_test)
```

👉 Predicts class labels

---

### 🔹 9. Probability Prediction

```python id="8j2l0n"
model.predict_proba(X_test)
```

👉 Returns probability of each class

---

## 📊 Evaluation Metrics

---

### 🔹 Accuracy

$$
Accuracy = \frac{Correct Predictions}{Total Predictions}
$$

---

### 🔹 Precision

$$
Precision = \frac{TP}{TP + FP}
$$

👉 Measures correctness of positive predictions

---

### 🔹 Recall

$$
Recall = \frac{TP}{TP + FN}
$$

👉 Measures ability to find all positives

---

### 🔹 F1 Score

$$
F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}
$$

👉 Balance between Precision & Recall

---

### 🔹 Confusion Matrix

|                 | Predicted Positive  | Predicted Negative  |
| --------------- | ------------------- | ------------------- |
| Actual Positive | True Positive (TP)  | False Negative (FN) |
| Actual Negative | False Positive (FP) | True Negative (TN)  |

---

### 🔹 ROC Curve & AUC

* ROC Curve → Performance at different thresholds
* AUC → Area under curve (higher = better model)

---

## 📈 Visualizations Used

* 📊 Correlation Heatmap
* 📊 Class Distribution Plot
* 📊 Confusion Matrix Heatmap
* 📊 ROC Curve

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

```bash id="nq3c4v"
pip install numpy pandas matplotlib seaborn scikit-learn
```

2. Run script:

```bash id="j2d4x9"
python day4_logistic_regression.py
```

---

## 📊 Output

* Predicted class labels
* Probability scores
* Evaluation metrics (Accuracy, Precision, Recall, F1)
* Confusion Matrix visualization
* ROC Curve

---

## ⚠️ Important Interview Questions

### ❓ What is Logistic Regression?

A classification algorithm that predicts probability using a sigmoid function.

---

### ❓ Why use Sigmoid Function?

To convert output into probability (0 to 1).

---

### ❓ Difference: Linear vs Logistic Regression?

| Linear Regression          | Logistic Regression |
| -------------------------- | ------------------- |
| Predicts continuous values | Predicts categories |
| Uses straight line         | Uses sigmoid curve  |

---

### ❓ What is Precision vs Recall?

* Precision → Correct positive predictions
* Recall → Total actual positives detected

---

### ❓ What is ROC-AUC?

Measures model performance across all classification thresholds.

---

## 🎯 Conclusion

This project demonstrates:

* Binary classification using Logistic Regression
* Model evaluation using multiple metrics
* Visualization of classification performance

👉 Logistic Regression is a **core algorithm for classification problems**

---

## 🔥 Next Step

Move to **Day 5: KNN & Naive Bayes (Distance & Probability Models)** 🚀

---
