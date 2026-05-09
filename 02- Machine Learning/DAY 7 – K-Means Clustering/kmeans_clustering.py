
# ==============================
# 🔹 STEP 1: IMPORT LIBRARIES
# ==============================

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# ML Libraries
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==============================
# 🔹 STEP 2: CREATE SAMPLE DATASET
# ==============================

# Customer data
data = {
    'Annual_Income': [
        15, 16, 17, 18, 20,
        60, 62, 64, 65, 68,
        100, 102, 105, 108, 110
    ],

    'Spending_Score': [
        39, 40, 42, 45, 48,
        50, 55, 58, 60, 62,
        80, 82, 85, 88, 90
    ]
}

df = pd.DataFrame(data)

print("\n📌 Dataset:")
print(df)

# ==============================
# 🔹 STEP 3: VISUALIZE DATA
# ==============================

plt.figure(figsize=(8,5))

plt.scatter(
    df['Annual_Income'],
    df['Spending_Score']
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("📊 Customer Data")
plt.show()

# ==============================
# 🔹 STEP 4: FEATURE SCALING
# ==============================

scaler = StandardScaler()

scaled_data = scaler.fit_transform(df)

# ==============================
# 🔹 STEP 5: ELBOW METHOD
# ==============================

# Find best K value
wcss = []

for k in range(1, 11):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(scaled_data)

    wcss.append(kmeans.inertia_)

# Plot elbow graph
plt.figure(figsize=(8,5))

plt.plot(range(1,11), wcss, marker='o')

plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.title("📊 Elbow Method")
plt.show()

# ==============================
# 🔹 STEP 6: APPLY K-MEANS
# ==============================

# Choose K = 3
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

# Predict clusters
clusters = kmeans.fit_predict(scaled_data)

# Add cluster column
df['Cluster'] = clusters

print("\n📌 Clustered Data:")
print(df)

# ==============================
# 🔹 STEP 7: CLUSTER CENTERS
# ==============================

print("\n📌 Cluster Centers:")

centers = scaler.inverse_transform(kmeans.cluster_centers_)

print(centers)

# ==============================
# 🔹 STEP 8: VISUALIZE CLUSTERS
# ==============================

plt.figure(figsize=(8,5))

# Plot customers
plt.scatter(
    df['Annual_Income'],
    df['Spending_Score'],
    c=df['Cluster']
)

# Plot cluster centers
plt.scatter(
    centers[:,0],
    centers[:,1],
    s=300,
    marker='X'
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("📊 K-Means Clustering")
plt.show()

# ==============================
# 🔹 STEP 9: UNDERSTANDING CLUSTERS
# ==============================

for i in range(3):

    print(f"\n📌 Customers in Cluster {i}:")
    print(df[df['Cluster'] == i])

# ==============================
# 🔹 STEP 10: FINAL SUMMARY
# ==============================

print("\n🎯 DAY 7 COMPLETED!")
print("You implemented K-Means Clustering 🚀")

print("\n📌 Concepts Covered:")
print("- Unsupervised Learning")
print("- K-Means Clustering")
print("- Elbow Method")
print("- Cluster Visualization")
print("- Customer Segmentation")