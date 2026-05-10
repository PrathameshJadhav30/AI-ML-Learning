
# ==============================
# 🔹 STEP 1: IMPORT LIBRARIES
# ==============================

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# ML Libraries
from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==============================
# 🔹 STEP 2: LOAD DATASET
# ==============================

data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)

df['TARGET'] = data.target

print("\n📌 Dataset Preview:")
print(df.head())

# ==============================
# 🔹 STEP 3: FEATURE & TARGET
# ==============================

X = df.drop('TARGET', axis=1)
y = df['TARGET']

# ==============================
# 🔹 STEP 4: TRAIN-TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==============================
# 🔹 STEP 5: TRAIN MODEL
# ==============================

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

# ==============================
# 🔹 STEP 6: MAKE PREDICTIONS
# ==============================

y_pred = model.predict(X_test)

# ==============================
# 🔹 STEP 7: EVALUATION METRICS
# ==============================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

print("\n📊 MODEL EVALUATION")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

# ==============================
# 🔹 STEP 8: CONFUSION MATRIX
# ==============================

cm = confusion_matrix(y_test, y_pred)

print("\n📌 Confusion Matrix:")
print(cm)

# Visualize confusion matrix
plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("📊 Confusion Matrix")

plt.show()

# ==============================
# 🔹 STEP 9: CLASSIFICATION REPORT
# ==============================

print("\n📌 Classification Report:")
print(classification_report(y_test, y_pred))

# ==============================
# 🔹 STEP 10: CROSS VALIDATION
# ==============================

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5
)

print("\n📊 Cross Validation Scores:")
print(cv_scores)

print("\nAverage CV Score:", cv_scores.mean())

# ==============================
# 🔹 STEP 11: HYPERPARAMETER TUNING
# ==============================

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5, 7]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5
)

grid_search.fit(X_train, y_train)

# Best parameters
print("\n📌 Best Parameters:")
print(grid_search.best_params_)

# Best model
best_model = grid_search.best_estimator_

# ==============================
# 🔹 STEP 12: EVALUATE BEST MODEL
# ==============================

best_pred = best_model.predict(X_test)

best_accuracy = accuracy_score(y_test, best_pred)

print("\n📊 Best Model Accuracy:")
print(best_accuracy)

# ==============================
# 🔹 STEP 13: FEATURE IMPORTANCE
# ==============================

importance = pd.Series(
    best_model.feature_importances_,
    index=X.columns
)

importance = importance.sort_values(ascending=False)

print("\n📌 Feature Importance:")
print(importance)

# Plot feature importance
plt.figure(figsize=(10,5))

importance.head(10).plot(kind='bar')

plt.title("📊 Top 10 Important Features")
plt.ylabel("Importance Score")

plt.show()

# ==============================
# 🔹 STEP 14: MODEL COMPARISON
# ==============================

models = ['Original Model', 'Tuned Model']

accuracies = [
    accuracy,
    best_accuracy
]

plt.figure(figsize=(6,4))

plt.bar(models, accuracies)

plt.ylabel("Accuracy")
plt.title("📊 Model Comparison")

plt.show()

# ==============================
# 🔹 STEP 15: FINAL SUMMARY
# ==============================

print("\n🎯 DAY 8 COMPLETED!")
print("You implemented Model Evaluation & Hyperparameter Tuning 🚀")

print("\n📌 Concepts Covered:")
print("- Accuracy")
print("- Precision")
print("- Recall")
print("- F1 Score")
print("- Confusion Matrix")
print("- Cross Validation")
print("- GridSearchCV")
print("- Hyperparameter Tuning")