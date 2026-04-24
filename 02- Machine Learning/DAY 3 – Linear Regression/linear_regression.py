
# ==============================
# 🔹 STEP 1: IMPORT LIBRARIES
# ==============================

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ==============================
# 🔹 STEP 2: LOAD DATASET
# ==============================

from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()

df = pd.DataFrame(data.data, columns=data.feature_names)
df['PRICE'] = data.target

print("\n📌 Dataset Preview:\n", df.head())

# ==============================
# 🔹 STEP 3: DATA UNDERSTANDING
# ==============================

print("\n📌 Shape:", df.shape)
print("\n📌 Info:")
print(df.info())
print("\n📌 Description:")
print(df.describe())

# ==============================
# 🔹 STEP 4: VISUALIZATION
# ==============================

# Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("📊 Correlation Heatmap")
plt.show()

# Scatter plot (important feature vs target)
plt.figure()
plt.scatter(df['MedInc'], df['PRICE'])
plt.xlabel("Median Income")
plt.ylabel("House Price")
plt.title("📊 Income vs Price")
plt.show()

# ==============================
# 🔹 STEP 5: FEATURE & TARGET
# ==============================

X = df.drop('PRICE', axis=1)
y = df['PRICE']

# ==============================
# 🔹 STEP 6: TRAIN-TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\n📌 Train Shape:", X_train.shape)
print("📌 Test Shape:", X_test.shape)

# ==============================
# 🔹 STEP 7: MODEL TRAINING
# ==============================

model = LinearRegression()
model.fit(X_train, y_train)

print("\n✅ Model Trained Successfully!")

# ==============================
# 🔹 STEP 8: MODEL EQUATION
# ==============================

# y = m1x1 + m2x2 + ... + c
print("\n📌 Intercept (c):", model.intercept_)
print("\n📌 Coefficients (m):")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")

# ==============================
# 🔹 STEP 9: PREDICTIONS
# ==============================

y_pred = model.predict(X_test)

print("\n📌 Sample Predictions:\n", y_pred[:5])

# ==============================
# 🔹 STEP 10: MODEL EVALUATION
# ==============================

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Evaluation Metrics:")
print("Mean Squared Error:", mse)
print("R² Score:", r2)

# ==============================
# 🔹 STEP 11: ACTUAL vs PREDICTED
# ==============================

plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("📊 Actual vs Predicted")
plt.show()

# ==============================
# 🔹 STEP 12: RESIDUAL ANALYSIS
# ==============================

residuals = y_test - y_pred

plt.figure()
sns.histplot(residuals, bins=30)
plt.title("📊 Residual Distribution")
plt.show()

# ==============================
# 🔹 STEP 13: LINEAR RELATION CHECK
# ==============================

# Plot regression line (single feature for visualization)
plt.figure()
sns.regplot(x=df['MedInc'], y=df['PRICE'])
plt.title("📊 Regression Line (Income vs Price)")
plt.show()

# ==============================
# 🔹 STEP 14: FINAL SUMMARY
# ==============================

print("\n🎯 DAY 3 COMPLETED!")
print("You implemented Linear Regression end-to-end 🚀")

print("\n📌 Summary:")
print("- Loaded dataset")
print("- Visualized relationships")
print("- Trained Linear Regression model")
print("- Evaluated using MSE & R²")
print("- Checked residuals")