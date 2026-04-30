
# ==============================
# 🔹 STEP 1: IMPORT LIBRARIES
# ==============================

import pandas as pd

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==============================
# 🔹 STEP 2: LOAD DATASET
# ==============================

data = load_wine()

# Convert to DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df['TARGET'] = data.target

print("📌 Dataset Preview:")
print(df.head())

# ==============================
# 🔹 STEP 3: FEATURES & TARGET
# ==============================

X = df.drop('TARGET', axis=1)
y = df['TARGET']

# ==============================
# 🔹 STEP 4: TRAIN-TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 🔹 STEP 5: DECISION TREE
# ==============================

dt = DecisionTreeClassifier()

dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

dt_acc = accuracy_score(y_test, y_pred_dt)

print("\n🌳 Decision Tree Accuracy:", dt_acc)

# ==============================
# 🔹 STEP 6: RANDOM FOREST
# ==============================

rf = RandomForestClassifier()

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

rf_acc = accuracy_score(y_test, y_pred_rf)

print("\n🌲 Random Forest Accuracy:", rf_acc)

# ==============================
# 🔹 STEP 7: COMPARISON
# ==============================

print("\n📊 Model Comparison:")
print("Decision Tree:", dt_acc)
print("Random Forest:", rf_acc)

# ==============================
# 🔹 STEP 8: FEATURE IMPORTANCE
# ==============================

importances = pd.Series(rf.feature_importances_, index=X.columns)

print("\n📌 Feature Importance:")
print(importances.sort_values(ascending=False))

# ==============================
# 🔹 STEP 9: OVERFITTING CHECK
# ==============================

train_acc = dt.score(X_train, y_train)
test_acc = dt.score(X_test, y_test)

print("\n📌 Overfitting Check (Decision Tree):")
print("Training Accuracy:", train_acc)
print("Testing Accuracy:", test_acc)

# ==============================
# 🔹 FINAL
# ==============================

print("\n🎯 DAY 6 COMPLETED!")