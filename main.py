import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# Load the dataset
df = pd.read_csv('Spotify_Youtube.csv')
df.head()

# Preprocess
# Check for columns with null/na values
np.where(pd.isna(df))
# Check for missing values
df.isnull().sum()
# Check for duplicate rows
df.duplicated().sum()

# Focus on the following three columns: Liveness, Energy, Loudness
X = df[['Liveness', 'Energy', 'Loudness']].copy()
# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method to find the optimal number of K 
inertias = []
K_range = range(1, 15)
for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.figure()
plt.plot(K_range, inertias, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')

# Visualize in 3D
optimal_K = 5       # Determined based the Elbow diagram above

km = KMeans(n_clusters=optimal_K, n_init = 10, max_iter = 300, random_state=0)
y_km = km.fit_predict(X_scaled)

scaled_centers = km.cluster_centers_       #a list of cluster centers
df2 = scaled_centers.copy()
df2 = pd.DataFrame(df2,columns=['Liveness', 'Energy', 'Loudness'])
centers_original = scaler.inverse_transform(scaled_centers)
df3 = pd.DataFrame(centers_original,columns=['Liveness', 'Energy', 'Loudness'])

cluster_labels = {
    0: "Moderate Studio Recording",
    1: "High-Energy Studio Banger",
    2: "Quiet Ambient",
    3: "Live High-Energy Performance",
    4: "Hybrid Live-Studio Track",
}
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Create a list to store scatter plots for legend
scatters = []

# Plot each cluster with consistent colors
for i in range(optimal_K):
    scat = ax.scatter(X_scaled[y_km==i, 0], 
                      X_scaled[y_km==i, 1], 
                      X_scaled[y_km==i, 2], 
                      label=cluster_labels[i],
                      alpha=0.7)
    scatters.append(scat)

# Plot cluster centers
ax.scatter(km.cluster_centers_[:,0], 
           km.cluster_centers_[:,1], 
           km.cluster_centers_[:,2], 
           s=200, marker='*', c='black', label='Cluster Centers')

ax.set_title(f'3D Visualization of K-Means Clusters (K={optimal_K})')
ax.set_xlabel("Liveness (scaled)")
ax.set_ylabel("Energy (scaled)")
ax.set_zlabel("Loudness (scaled)")
ax.legend(bbox_to_anchor=(1.2, 1), loc='upper left', title="Clusters")

plt.tight_layout()
plt.show()

# Hierarchical Clustering
columns_to_cluster = ['Liveness', 'Energy', 'Loudness']

# Hierarchical Clustering for all - Dendrogram
all = linkage(X_scaled, method='ward')
plt.figure(figsize=(12, 6))
dendrogram(all)
plt.title(f'Hierarchical Clustering Dendrogram')
plt.xlabel('Sample')
plt.ylabel('Distance')
plt.show()

# Hierarchical Clustering for each individual column - Dendrogram (will determine the number of clusters)
column_indices = [X.columns.get_loc(col) for col in columns_to_cluster]
for i, column in enumerate(columns_to_cluster):
    # Extract the column data and reshape it for clustering (-1,1 makes it 2D)
    X_col = X_scaled[:, column_indices[i]].reshape(-1, 1)  # Reshape to (n_samples, 1)
    
    # Perform hierarchical clustering
    linked = linkage(X_col, method='ward')
    
    # Plot dendrogram
    plt.figure(figsize=(12, 6))
    dendrogram(linked,
               orientation='top',
               distance_sort='descending',
               show_leaf_counts=True)
    plt.title(f'Hierarchical Clustering Dendrogram for {column}')
    plt.xlabel('Sample')
    plt.ylabel('Distance (Ward)')
    plt.show()

# Hierarchical Clustering for each individual column
for i, column in enumerate(columns_to_cluster):
    # Extract the column data and reshape it for clustering (-1,1 makes it 2D)
    X_col = X_scaled[:, column_indices[i]].reshape(-1, 1)  # Reshape to (n_samples, 1)
        
    # Determine number of clusters based on the individual dendrogram above, the reasonable choice is 3 clusters
    n_clusters = 3
    
    # Perform Agglomerative Clustering (updated parameter)
    cluster = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
    cluster_labels = cluster.fit_predict(X_col)
    
    # Plot the clusters
    plt.figure(figsize=(10, 6))
    for i in range(n_clusters):
        plt.scatter(np.arange(len(X_col))[cluster_labels == i], 
                   X_col[cluster_labels == i], 
                   label=f'Cluster {i+1}')
    
    plt.title(f'Hierarchical Clustering Results for {column}')
    plt.xlabel('Sample Index')
    plt.ylabel(column)
    plt.legend()
    plt.show()
    
    # Print cluster statistics
    print(f"\nCluster statistics for {column}:")
    for i in range(n_clusters):
        cluster_data = X_col[cluster_labels == i]
        print(f"Cluster {i+1}:")
        print(f"  Size: {len(cluster_data)}")
        print(f"  Min: {cluster_data.min():.3f}")
        print(f"  Max: {cluster_data.max():.3f}")
        print(f"  Mean: {cluster_data.mean():.3f}")
        print(f"  Std: {cluster_data.std():.3f}")
        print()