
# ==============================
# 🔹 STEP 1: IMPORT LIBRARIES
# ==============================

import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

# ==============================
# 🔹 STEP 2: CREATE SAMPLE DATASET
# ==============================

# Creating dataset manually
data = {
    'Age': [25, 30, 35, np.nan, 28, 40, 50, np.nan],
    'Salary': [50000, 60000, 65000, 70000, np.nan, 80000, 90000, 100000],
    'City': ['Pune', 'Mumbai', 'Delhi', 'Pune', 'Delhi', 'Mumbai', 'Pune', 'Delhi'],
    'Purchased': ['No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

print("\n📌 Original Dataset:\n", df)

# ==============================
# 🔹 STEP 3: HANDLE MISSING VALUES
# ==============================

print("\n📌 Missing Values Before:\n", df.isnull().sum())

# Fill missing numerical values with mean
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Salary'].fillna(df['Salary'].mean(), inplace=True)

print("\n📌 Missing Values After:\n", df.isnull().sum())

# ==============================
# 🔹 STEP 4: ENCODING CATEGORICAL DATA
# ==============================

# 🔸 Label Encoding (for target variable)
label_encoder = LabelEncoder()
df['Purchased'] = label_encoder.fit_transform(df['Purchased'])

print("\n📌 After Label Encoding (Purchased):\n", df)

# ==============================
# 🔹 STEP 5: FEATURE & TARGET SPLIT
# ==============================

X = df[['Age', 'Salary', 'City']]
y = df['Purchased']

# ==============================
# 🔹 STEP 6: TRAIN-TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\n📌 Training Data:\n", X_train)
print("\n📌 Testing Data:\n", X_test)

# ==============================
# 🔹 STEP 7: ONE-HOT ENCODING (CITY)
# ==============================

# Apply OneHotEncoding to 'City'
ct = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(), ['City'])
    ],
    remainder='passthrough'
)

X_train = ct.fit_transform(X_train)
X_test = ct.transform(X_test)

print("\n📌 After One-Hot Encoding:\n", X_train)

# ==============================
# 🔹 STEP 8: FEATURE SCALING
# ==============================

scaler = StandardScaler()

# Scale numerical features
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\n📌 After Feature Scaling:\n", X_train)

# ==============================
# 🔹 STEP 9: VISUALIZATION (OPTIONAL)
# ==============================

# Convert processed data to DataFrame for visualization
processed_df = pd.DataFrame(X_train)

plt.figure(figsize=(8,5))
sns.heatmap(processed_df.corr(), annot=True, cmap='coolwarm')
plt.title("📊 Correlation After Preprocessing")
plt.show()

# ==============================
# 🔹 STEP 10: FINAL OUTPUT
# ==============================

print("\n🎯 DAY 2 COMPLETED SUCCESSFULLY!")
print("You learned Data Preprocessing 🚀")

# Summary
print("\n📌 Summary:")
print("- Missing values handled")
print("- Categorical data encoded")
print("- Features scaled")
print("- Data ready for ML model")