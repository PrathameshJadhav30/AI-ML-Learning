
# ==============================
# 🔹 STEP 1: IMPORT LIBRARIES
# ==============================

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# ==============================
# 🔹 STEP 2: LOAD DATASET
# ==============================

# Using Breast Cancer dataset (binary classification)
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)
df['TARGET'] = data.target  # 0 = malignant, 1 = benign

print("\n📌 Dataset Preview:\n", df.head())

# ==============================
# 🔹 STEP 3: DATA UNDERSTANDING
# ==============================

print("\n📌 Shape:", df.shape)
print("\n📌 Info:")
print(df.info())

print("\n📌 Class Distribution:\n", df['TARGET'].value_counts())

# ==============================
# 🔹 STEP 4: VISUALIZATION
# ==============================

# Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("📊 Correlation Heatmap")
plt.show()

# Count plot (class distribution)
plt.figure()
sns.countplot(x=df['TARGET'])
plt.title("📊 Class Distribution")
plt.show()

# ==============================
# 🔹 STEP 5: FEATURE & TARGET
# ==============================

X = df.drop('TARGET', axis=1)
y = df['TARGET']

# ==============================
# 🔹 STEP 6: TRAIN-TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ==============================
# 🔹 STEP 7: FEATURE SCALING
# ==============================

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# 🔹 STEP 8: MODEL TRAINING
# ==============================

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("\n✅ Logistic Regression Model Trained!")

# ==============================
# 🔹 STEP 9: PREDICTIONS
# ==============================

y_pred = model.predict(X_test)

# Probability predictions (Sigmoid output)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n📌 Sample Predictions:", y_pred[:5])
print("\n📌 Prediction Probabilities:", y_prob[:5])

# ==============================
# 🔹 STEP 10: EVALUATION METRICS
# ==============================

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n📊 Model Evaluation:")
print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1 Score:", f1)

# ==============================
# 🔹 STEP 11: CONFUSION MATRIX
# ==============================

cm = confusion_matrix(y_test, y_pred)

print("\n📌 Confusion Matrix:\n", cm)

# Visualization of confusion matrix
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("📊 Confusion Matrix")
plt.show()

# ==============================
# 🔹 STEP 12: CLASSIFICATION REPORT
# ==============================

print("\n📌 Classification Report:\n")
print(classification_report(y_test, y_pred))

# ==============================
# 🔹 STEP 13: ROC CURVE
# ==============================

from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("📊 ROC Curve")
plt.legend()
plt.show()

# ==============================
# 🔹 STEP 14: FINAL SUMMARY
# ==============================

print("\n🎯 DAY 4 COMPLETED!")
print("You implemented Logistic Regression end-to-end 🚀")

print("\n📌 Summary:")
print("- Binary Classification performed")
print("- Model trained using Logistic Regression")
print("- Evaluated using Accuracy, Precision, Recall, F1")
print("- Confusion Matrix & ROC Curve analyzed")