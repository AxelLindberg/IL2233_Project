import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from minisom import MiniSom  # Install using: pip install minisom

# Generate bivariate time series data
np.random.seed(42)
X1 = np.random.normal(loc=0, scale=2, size=200)
X2 = np.random.normal(loc=1, scale=2, size=200)

# Visualize the data
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(X1)
plt.title('X1 Time Series')
plt.subplot(1, 2, 2)
plt.plot(X2)
plt.title('X2 Time Series')
plt.show()

plt.figure(figsize=(7, 7))
plt.scatter(X1, X2)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('X1 vs X2 Scatter Plot')
plt.show()


# Combine X1 and X2 into a single dataset
data = np.column_stack((X1, X2))

# Perform K-means clustering
kmeans = KMeans(n_clusters=2)  # You may need to adjust the number of clusters
kmeans.fit(data)

# Calculate distances to centroids
distances = kmeans.transform(data)

# Sort data points based on distances
sorted_indices = np.argsort(np.max(distances, axis=1))[::-1]

# Identify anomalies based on specified anomaly ratio
anomaly_ratio = 0.02
num_anomalies = int(anomaly_ratio * len(data))
anomaly_indices = sorted_indices[:num_anomalies]
anomalies = data[anomaly_indices]

# Visualize anomalies
plt.figure(figsize=(7, 7))
plt.scatter(X1, X2, label='Normal Data')
plt.scatter(anomalies[:, 0], anomalies[:, 1], color='red', label='Anomalies')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Anomaly Detection with K-means')
plt.legend()
plt.show()


# Initialize SOM
som = MiniSom(5, 5, 2, sigma=0.5, learning_rate=0.5)
som.random_weights_init(data)
som.train_random(data, 100)  # Train SOM

# Find anomalies with SOM
winner_coordinates = np.array([som.winner(x) for x in data])
distances = np.linalg.norm(data - som.get_weights()[winner_coordinates[:, 0], winner_coordinates[:, 1]], axis=1)
sorted_indices = np.argsort(distances)[::-1]
anomaly_indices_som = sorted_indices[:num_anomalies]
anomalies_som = data[anomaly_indices_som]

# Visualize anomalies with SOM
plt.figure(figsize=(7, 7))
plt.scatter(X1, X2, label='Normal Data')
plt.scatter(anomalies_som[:, 0], anomalies_som[:, 1], color='red', label='Anomalies')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Anomaly Detection with SOM')
plt.legend()
plt.show()

