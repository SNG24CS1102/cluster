import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("Libraries imported successfully.")


# Step 2: Load and preprocess the Digits dataset
digits = load_digits()

X = digits.data
y = digits.target

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Dataset successfully loaded: {X.shape[0]} samples")
print(f"Number of features: {X.shape[1]}")
print(f"Number of actual digit classes: {len(np.unique(y))}")


# Step 3: Find the optimal number of clusters
cluster_range = range(2, 21)

inertia_scores = []
silhouette_scores = []

print(f"\n{'K (Clusters)':<12} | {'Inertia':<12} | {'Silhouette Score':<16}")
print("-" * 48)

for k in cluster_range:

    kmeans = KMeans(
        n_clusters=k,
        init='k-means++',
        random_state=42,
        n_init=10
    )

    cluster_labels = kmeans.fit_predict(X_scaled)

    inertia = kmeans.inertia_
    silhouette = silhouette_score(X_scaled, cluster_labels)

    inertia_scores.append(inertia)
    silhouette_scores.append(silhouette)

    print(f"{k:<12} | {inertia:<12.2f} | {silhouette:<16.4f}")


# Step 4: Visualize the Elbow method and Silhouette score
plt.figure(figsize=(15, 5))

# Elbow method
plt.subplot(1, 2, 1)

plt.plot(
    cluster_range,
    inertia_scores,
    marker='o',
    linestyle='-'
)

plt.title('Elbow Analysis: Inertia Reduction Plot')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (WCSS)')
plt.xticks(list(cluster_range))
plt.grid(True, linestyle=':', alpha=0.6)


# Silhouette score
plt.subplot(1, 2, 2)

plt.plot(
    cluster_range,
    silhouette_scores,
    marker='s',
    linestyle='-'
)

plt.title('Silhouette Score Across Cluster Quantities')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.xticks(list(cluster_range))
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()


# Step 5: Apply K-Means with 10 clusters
optimal_k = 10

kmeans_optimal = KMeans(
    n_clusters=optimal_k,
    init='k-means++',
    random_state=42,
    n_init=10
)

optimal_labels = kmeans_optimal.fit_predict(X_scaled)

print(f"\nK-Means clustering completed with K = {optimal_k}")


# Step 6: Reduce dimensions using PCA
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)


# Step 7: Visualize the clusters
plt.figure(figsize=(9, 6))

scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=optimal_labels,
    cmap='tab10',
    alpha=0.6,
    s=15
)

plt.colorbar(scatter, label='Cluster ID')

plt.title(f'PCA Cluster Visualization (K={optimal_k})')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True, linestyle=':', alpha=0.5)

plt.show()


# Step 8: Display final results
print("\nFinal Results")
print("-----------------------------")
print("Total samples:", X.shape[0])
print("Total features:", X.shape[1])
print("Number of clusters:", optimal_k)
print("Final inertia:", round(kmeans_optimal.inertia_, 2))
print(
    "Silhouette score:",
    round(silhouette_score(X_scaled, optimal_labels), 4)
)