import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


def generate_birch_data(grid_size=10, center_distance=4*np.sqrt(2), points_per_cluster=100):
    xs = [i*center_distance for i in range(grid_size)]
    ys = [i*center_distance for i in range(grid_size)]
    xs, ys = np.meshgrid(xs, ys)
    cluster_centers = np.hstack((np.expand_dims(np.ravel(xs), axis=-1), np.expand_dims(np.ravel(ys), axis=-1)))

    num_clusters = grid_size * grid_size
    num_samples = num_clusters * points_per_cluster

    data, labels = make_blobs(num_samples, centers=cluster_centers, random_state=0)

    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)
    normalized_centers = scaler.transform(cluster_centers)

    return normalized_data, labels, normalized_centers