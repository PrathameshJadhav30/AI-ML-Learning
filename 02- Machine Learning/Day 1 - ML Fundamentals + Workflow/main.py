# ============================================
# 🚀 DAY 1: MACHINE LEARNING FULL IMPLEMENTATION
# ============================================

# ==============================
# 🔹 STEP 1: IMPORT LIBRARIES
# ==============================

# Numerical operations
import numpy as np

# Data handling
import pandas as pd

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning utilities
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ==============================
# 🔹 STEP 2: LOAD DATASET
# ==============================

# Import dataset from sklearn
from sklearn.datasets import fetch_california_housing

# Load dataset
data = fetch_california_housing()

# Convert into DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)

# Add target column (PRICE)
df['PRICE'] = data.target

# Display first 5 rows
print("\n📌 First 5 Rows of Dataset:\n", df.head())

# ==============================
# 🔹 STEP 3: DATA UNDERSTANDING
# ==============================

# Shape of dataset
print("\n📌 Dataset Shape:", df.shape)

# Information about dataset
print("\n📌 Dataset Info:")
print(df.info())

# Statistical summary
print("\n📌 Statistical Summary:")
print(df.describe())

# ==============================
# 🔹 STEP 4: CHECK MISSING VALUES
# ==============================

print("\n📌 Missing Values:\n", df.isnull().sum())

# If missing values exist → fill with mean
df.fillna(df.mean(), inplace=True)

# ==============================
# 🔹 STEP 5: DATA VISUALIZATION
# ==============================

# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("📊 Feature Correlation Heatmap")
plt.show()

# Distribution of target variable
plt.figure()
sns.histplot(df['PRICE'], bins=30)
plt.title("📊 Price Distribution")
plt.show()

# Scatter plot (example relationship)
plt.figure()
plt.scatter(df['MedInc'], df['PRICE'])
plt.xlabel("Median Income")
plt.ylabel("House Price")
plt.title("📊 Income vs Price")
plt.show()

# ==============================
# 🔹 STEP 6: FEATURE & TARGET
# ==============================

# Features (independent variables)
X = df.drop('PRICE', axis=1)

# Target (dependent variable)
y = df['PRICE']

# ==============================
# 🔹 STEP 7: TRAIN-TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 80% training, 20% testing
    random_state=42      # For reproducibility
)

print("\n📌 Training Data Shape:", X_train.shape)
print("📌 Testing Data Shape:", X_test.shape)

# ==============================
# 🔹 STEP 8: MODEL TRAINING
# ==============================

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

print("\n✅ Model Training Completed!")

# ==============================
# 🔹 STEP 9: MAKE PREDICTIONS
# ==============================

# Predict on test data
y_pred = model.predict(X_test)

print("\n📌 Sample Predictions:\n", y_pred[:5])

# ==============================
# 🔹 STEP 10: MODEL EVALUATION
# ==============================

# Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

# R² Score
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation:")
print("Mean Squared Error:", mse)
print("R² Score:", r2)

# ==============================
# 🔹 STEP 11: ACTUAL vs PREDICTED
# ==============================

plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual vs Predicted Prices")
plt.show()

#  STEP 12: MODEL INSIGHTS
# Model coefficients
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])

print("\n Feature Importance (Coefficients):\n", coefficients)

# STEP 13: FINAL OUTPUT

print("\n DAY 1 COMPLETED SUCCESSFULLY!")
print("You have built your first Machine Learning model ")