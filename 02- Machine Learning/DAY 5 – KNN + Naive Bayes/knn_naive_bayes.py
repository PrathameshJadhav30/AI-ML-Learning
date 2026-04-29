
# ==============================
# 🔹 STEP 1: IMPORT LIBRARIES
# ==============================

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# ==============================
# 🔹 STEP 2: LOAD DATASET
# ==============================

from sklearn.datasets import load_iris

data = load_iris()

df = pd.DataFrame(data.data, columns=data.feature_names)
df['TARGET'] = data.target

print("\n📌 Dataset Preview:\n", df.head())

# ==============================
# 🔹 STEP 3: DATA UNDERSTANDING
# ==============================

print("\n📌 Shape:", df.shape)
print("\n📌 Class Distribution:\n", df['TARGET'].value_counts())

# ==============================
# 🔹 STEP 4: VISUALIZATION
# ==============================

sns.pairplot(df, hue='TARGET')
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
# 🔹 STEP 7: FEATURE SCALING (IMPORTANT FOR KNN)
# ==============================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# 🔹 STEP 8: KNN MODEL
# ==============================

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

# Evaluation
print("\n📊 KNN Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
print("Classification Report:\n", classification_report(y_test, y_pred_knn))

# ==============================
# 🔹 STEP 9: NAIVE BAYES MODEL
# ==============================

nb = GaussianNB()

nb.fit(X_train, y_train)

y_pred_nb = nb.predict(X_test)

# Evaluation
print("\n📊 Naive Bayes Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_nb))
print("Classification Report:\n", classification_report(y_test, y_pred_nb))

# ==============================
# 🔹 STEP 10: MODEL COMPARISON
# ==============================

knn_acc = accuracy_score(y_test, y_pred_knn)
nb_acc = accuracy_score(y_test, y_pred_nb)

print("\n📊 Model Comparison:")
print("KNN Accuracy:", knn_acc)
print("Naive Bayes Accuracy:", nb_acc)

# ==============================
# 🔹 STEP 11: VISUALIZE ACCURACY
# ==============================

models = ['KNN', 'Naive Bayes']
accuracies = [knn_acc, nb_acc]

plt.figure()
plt.bar(models, accuracies)
plt.title("📊 Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.show()

# ==============================
# 🔹 STEP 12: K VALUE TUNING (KNN)
# ==============================

error_rates = []

for k in range(1, 15):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    pred_k = knn_model.predict(X_test)
    error = np.mean(pred_k != y_test)
    error_rates.append(error)

plt.figure()
plt.plot(range(1, 15), error_rates, marker='o')
plt.title("📊 Error Rate vs K Value")
plt.xlabel("K")
plt.ylabel("Error Rate")
plt.show()

# ==============================
# 🔹 STEP 13: FINAL SUMMARY
# ==============================

print("\n🎯 DAY 5 COMPLETED!")
print("You implemented KNN and Naive Bayes 🚀")

print("\n📌 Summary:")
print("- KNN (distance-based model)")
print("- Naive Bayes (probability-based model)")
print("- Model comparison done")
print("- Hyperparameter tuning (K value)")