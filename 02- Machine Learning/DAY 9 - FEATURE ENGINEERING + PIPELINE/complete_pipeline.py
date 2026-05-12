# =========================================================
# 🚀 DAY 9: FEATURE ENGINEERING + PIPELINE
# =========================================================

# =========================================================
# 🔹 STEP 1: IMPORT LIBRARIES
# =========================================================

import pandas as pd
import numpy as np

# Train test split
from sklearn.model_selection import train_test_split

# Missing value handling
from sklearn.impute import SimpleImputer

# Encoding & Scaling
from sklearn.preprocessing import (
    OneHotEncoder,
    LabelEncoder,
    StandardScaler,
    MinMaxScaler
)

# Column transformer
from sklearn.compose import ColumnTransformer

# Pipeline
from sklearn.pipeline import Pipeline

# Feature Selection
from sklearn.feature_selection import SelectKBest, chi2

# Model
from sklearn.ensemble import RandomForestClassifier

# Evaluation
from sklearn.metrics import accuracy_score

# =========================================================
# 🔹 STEP 2: CREATE DATASET
# =========================================================

data = {

    'Age': [25, 30, np.nan, 40, 45, 28],

    'Salary': [50000, 60000, 65000,
               np.nan, 90000, 52000],

    'City': ['Pune', 'Mumbai',
             'Delhi', 'Pune',
             'Delhi', 'Mumbai'],

    'Gender': ['Male', 'Female',
               'Male', 'Female',
               'Male', 'Female'],

    'Purchased': [0, 1, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

print("\n📌 ORIGINAL DATASET")
print(df)

# =========================================================
# 🔹 STEP 3: FEATURE ENGINEERING
# =========================================================

# Create new feature
df['Salary_Per_Age'] = df['Salary'] / (df['Age'] + 1)

print("\n📌 AFTER FEATURE ENGINEERING")
print(df)

# =========================================================
# 🔹 STEP 4: FEATURES & TARGET
# =========================================================

X = df.drop('Purchased', axis=1)

y = df['Purchased']

# =========================================================
# 🔹 STEP 5: IDENTIFY COLUMN TYPES
# =========================================================

numeric_features = [
    'Age',
    'Salary',
    'Salary_Per_Age'
]

categorical_features = [
    'City',
    'Gender'
]

# =========================================================
# 🔹 STEP 6: HANDLE NUMERIC DATA
# =========================================================

numeric_pipeline = Pipeline([

    # Fill missing values
    ('imputer', SimpleImputer(strategy='mean')),

    # Standard Scaling
    ('scaler', StandardScaler())
])

# =========================================================
# 🔹 STEP 7: HANDLE CATEGORICAL DATA
# =========================================================

categorical_pipeline = Pipeline([

    # Fill missing values
    ('imputer', SimpleImputer(strategy='most_frequent')),

    # Convert text → numbers
    ('encoder', OneHotEncoder())
])

# =========================================================
# 🔹 STEP 8: COMBINE BOTH PIPELINES
# =========================================================

preprocessor = ColumnTransformer([

    ('num', numeric_pipeline, numeric_features),

    ('cat', categorical_pipeline, categorical_features)
])

# =========================================================
# 🔹 STEP 9: FEATURE SELECTION
# =========================================================

feature_selection = SelectKBest(

    score_func=chi2,

    k=5
)

# =========================================================
# 🔹 STEP 10: CREATE COMPLETE PIPELINE
# =========================================================

pipeline = Pipeline([

    # Data preprocessing
    ('preprocessing', preprocessor),

    # Convert values between 0 and 1
    ('minmax', MinMaxScaler()),

    # Feature selection
    ('feature_selection', feature_selection),

    # ML model
    ('classifier', RandomForestClassifier(
        random_state=42
    ))
])

# =========================================================
# 🔹 STEP 11: TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42
)

# =========================================================
# 🔹 STEP 12: TRAIN MODEL
# =========================================================

pipeline.fit(X_train, y_train)

print("\n✅ PIPELINE TRAINED SUCCESSFULLY!")

# =========================================================
# 🔹 STEP 13: MAKE PREDICTIONS
# =========================================================

y_pred = pipeline.predict(X_test)

print("\n📌 PREDICTIONS")
print(y_pred)

# =========================================================
# 🔹 STEP 14: MODEL ACCURACY
# =========================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n📊 MODEL ACCURACY")
print(accuracy)

# =========================================================
# 🔹 STEP 15: LABEL ENCODING EXAMPLE
# =========================================================

print("\n📌 LABEL ENCODING EXAMPLE")

label_encoder = LabelEncoder()

df['Gender_Encoded'] = label_encoder.fit_transform(
    df['Gender']
)

print(df[['Gender', 'Gender_Encoded']])

# =========================================================
# 🔹 STEP 16: MANUAL SCALING EXAMPLE
# =========================================================

print("\n📌 STANDARD SCALER EXAMPLE")

scaler = StandardScaler()

scaled_salary = scaler.fit_transform(

    df[['Salary']].fillna(
        df['Salary'].mean()
    )
)

print(scaled_salary)

# =========================================================
# 🔹 STEP 17: MINMAX SCALER EXAMPLE
# =========================================================

print("\n📌 MINMAX SCALER EXAMPLE")

minmax = MinMaxScaler()

minmax_salary = minmax.fit_transform(

    df[['Salary']].fillna(
        df['Salary'].mean()
    )
)

print(minmax_salary)

# =========================================================
# 🔹 STEP 18: HANDLE MISSING VALUES MANUALLY
# =========================================================

print("\n📌 HANDLE MISSING VALUES")

df['Age'].fillna(

    df['Age'].mean(),

    inplace=True
)

df['Salary'].fillna(

    df['Salary'].mean(),

    inplace=True
)

print(df)

# =========================================================
# 🔹 STEP 19: SHOW PIPELINE STEPS
# =========================================================

print("\n📌 PIPELINE STEPS")

for name, step in pipeline.named_steps.items():

    print(name, " --> ", step)

# =========================================================
# 🔹 STEP 20: FINAL SUMMARY
# =========================================================

print("\n🎯 DAY 9 COMPLETED!")

print("\n📌 CONCEPTS IMPLEMENTED")

print("✅ Feature Engineering")
print("✅ Missing Value Handling")
print("✅ OneHotEncoder")
print("✅ LabelEncoder")
print("✅ StandardScaler")
print("✅ MinMaxScaler")
print("✅ ColumnTransformer")
print("✅ Pipeline")
print("✅ Feature Selection")
print("✅ Random Forest")
print("✅ Model Training")
print("✅ Prediction")
print("✅ Accuracy Evaluation")