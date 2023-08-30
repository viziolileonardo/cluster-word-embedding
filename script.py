import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances_argmin_min
import json

# Create an 'images' directory if it doesn't exist
if not os.path.exists('images'):
    os.makedirs('images')

# Load the DataFrame with embeddings
df = pd.read_csv('YOUROUTPUT.csv')

# Convert string representations of lists to actual lists
embedding_columns = [col for col in df.columns if col.startswith('embedding_')]
df[embedding_columns] = df[embedding_columns].applymap(lambda x: json.loads(x) if isinstance(x, str) else x)

# Drop any rows with NaN values in the embedding columns
df.dropna(subset=embedding_columns, inplace=True)

# Get the matrix of embeddings
matrix = np.vstack(df[embedding_columns].values)

# Loop over a range of cluster numbers for k-means
nrange = range (2, 10)
for n_clusters in nrange:
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
    kmeans.fit(matrix)
    labels = kmeans.labels_

    # Perform t-SNE for visualization
    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init="random", learning_rate=200)
    vis_dims2 = tsne.fit_transform(matrix)

    x = [x for x, y in vis_dims2]
    y = [y for x, y in vis_dims2]

    plt.figure(figsize=(15, 12))

    # Get closest points to centroids for semantic understanding
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, matrix)
    closest_points = [df['Internal CR Agents Notes'].iloc[idx] for idx in closest]

    for category in range(n_clusters):
        color = plt.cm.jet(float(category) / (n_clusters - 1))
        xs = np.array(x)[labels == category]
        ys = np.array(y)[labels == category]
        
        # Use the closest point as the label for the legend
        plt.scatter(xs, ys, color=color, alpha=0.3, label=f'Cluster {category}: {closest_points[category]}')
        
        avg_x = xs.mean()
        avg_y = ys.mean()
        plt.scatter(avg_x, avg_y, marker="x", color=color, s=100)

    plt.title(f"Clusters identified visualized in 2D using t-SNE (n_clusters={n_clusters})")
    plt.legend(title='Clusters (closest point)', loc='upper center')

    plt.savefig(os.path.join('images', f'clusters_{n_clusters}.png'))
    plt.close()
