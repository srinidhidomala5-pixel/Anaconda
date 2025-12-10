#!/usr/bin/env python
# coding: utf-8

# In[2]:


# -------------------------------
# IMPORT LIBRARIES
# -------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
#warnings.filterwarnings("ignore")

# -------------------------------
# STEP 1: LOAD DATA
# -------------------------------
df = pd.read_csv("Mall_Customers.csv")
print("Sample data:")
print(df.head())

# -------------------------------
# STEP 2: PREPROCESSING
# -------------------------------
# Use only numeric features for clustering
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# STEP 3: FIND OPTIMAL NUMBER OF CLUSTERS (ELBOW METHOD)
# -------------------------------
inertia = []
K = range(1,11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(8,5))
plt.plot(K, inertia, 'bo-')
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method For Optimal k")
plt.show()

# -------------------------------
# STEP 4: APPLY K-MEANS
# -------------------------------
# Choose k=5 (or based on elbow plot)
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster info to original data
df['Cluster'] = clusters

# -------------------------------
# STEP 5: VISUALIZATION
# -------------------------------
# 3D scatter plot
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

colors = ['red', 'blue', 'green', 'cyan', 'magenta']
for i in range(k):
    ax.scatter(
        df[df['Cluster']==i]['Age'],
        df[df['Cluster']==i]['Annual Income (k$)'],
        df[df['Cluster']==i]['Spending Score (1-100)'],
        s=50,
        c=colors[i],
        label=f'Cluster {i}'
    )

ax.set_xlabel("Age")
ax.set_ylabel("Annual Income (k$)")
ax.set_zlabel("Spending Score")
ax.set_title("Customer Segments")
ax.legend()
plt.show()

# 2D Visualization (Income vs Spending Score)
plt.figure(figsize=(8,6))
sns.scatterplot(
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Cluster',
    palette=colors,
    data=df,
    s=100
)
plt.title("Customer Segments (2D)")
plt.show()

# -------------------------------
# STEP 6: INTERPRETATION
# -------------------------------
for i in range(k):
    cluster_data = df[df['Cluster']==i]
    print(f"\nCluster {i} statistics:")
    print(cluster_data[['Age','Annual Income (k$)','Spending Score (1-100)']].describe())


# In[ ]:




