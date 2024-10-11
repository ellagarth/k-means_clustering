# Import necessary libraries
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('fake_student_dataset.csv')

# Feature selection
X = df[['GPA', 'Attendance_Rate', 'Study_Hours']]

# Options for user to modify the model's features
use_scaling = True  # Set to True to apply feature scaling
use_pca = False     # Set to True to apply PCA for dimensionality reduction
use_dbscan = False  # Set to True to use DBSCAN instead of K-Means
num_clusters = 5    # Number of clusters for K-Means
pca_components = 2  # Number of components for PCA, if applied

# Apply feature scaling if needed
if use_scaling:
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

# Apply PCA if needed
if use_pca:
    pca = PCA(n_components=pca_components)
    X = pca.fit_transform(X)

# Perform clustering: K-Means or DBSCAN
if use_dbscan:
    # DBSCAN clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    df['Cluster'] = dbscan.fit_predict(X)
else:
    # K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', n_init=10, random_state=42, max_iter=300)
    df['Cluster'] = kmeans.fit_predict(X)

# Save the dataset with the cluster labels
df.to_csv('student_clusters_with_options.csv', index=False)

# Visualize the clusters using GPA and Study Hours (if using K-Means or DBSCAN with PCA)
if not use_pca:
    plt.figure(figsize=(10, 6))
    plt.scatter(df['GPA'], df['Study_Hours'], c=df['Cluster'], cmap='viridis', marker='o')
    plt.title(f'Clustering: GPA vs Study Hours')
    plt.xlabel('GPA')
    plt.ylabel('Study Hours')
    plt.colorbar(label='Cluster')
    plt.show()

# Evaluation of Clustering Model
if not use_dbscan:
    # 1. Elbow Method (only for K-Means)
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    # Plot the Elbow graph
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), wcss, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS (Within-cluster Sum of Squares)')
    plt.show()

# 2. Silhouette Score
sil_score = silhouette_score(X, df['Cluster'])
print(f'Silhouette Score: {sil_score}')

# 3. Davies-Bouldin Index
db_index = davies_bouldin_score(X, df['Cluster'])
print(f'Davies-Bouldin Index: {db_index}')

# 4. Calinski-Harabasz Index
ch_score = calinski_harabasz_score(X, df['Cluster'])
print(f'Calinski-Harabasz Index: {ch_score}')
