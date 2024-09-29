"""Testing cv"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

import data_model

dm = data_model.DataModels()
X = dm.faces_vectors_x
print(X)

# Set min_samples (same as the min_samples parameter in DBSCAN)
min_samples = 5

# Compute the k-nearest neighbors
nearest_neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors = nearest_neighbors.fit(X)
distances, indices = neighbors.kneighbors(X)

# Sort the distances and plot the k-distance graph
distances = np.sort(distances[:, -1])  # Get the k-th nearest neighbor distance for each point
plt.plot(distances)
plt.ylabel('k-distance')
plt.xlabel('Data points (sorted by distance)')
plt.title('K-distance plot for DBSCAN')
plt.show()

# Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5, metric='euclidean')
clusters = dbscan.fit_predict(X)

# Output cluster labels (-1 indicates noise/outliers, 0, 1, 2... are cluster labels)
print("Cluster labels for each face:", clusters)

# You can group the faces by their cluster labels
# For example, group faces into dictionaries by cluster label:
clustered_faces:dict = {}
for i, label in enumerate(clusters):
    if label != -1:  # Ignore noise/outliers
        if label not in clustered_faces:
            clustered_faces[label] = []
        clustered_faces[label].append(X[i])
    else:
        print(f"Face {i+1} identified as noise/outlier")

print(f"Found {len(set(clusters)) - (1 if -1 in clusters else 0)} clusters")
