import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


def sample_unit_hypercube(dimensions, num_samples):
    samples = np.zeros((num_samples, dimensions))
    for i in range(dimensions):
        samples[:, i] = np.random.uniform(-0.5, 0.5, num_samples)
    return samples


def generate_pelleg_moore_data(dimensions=2, num_clusters=50, samples_per_cluster=50, cluster_std_factor=0.012):
    cluster_centers = sample_unit_hypercube(dimensions, num_clusters)
    num_samples = num_clusters * samples_per_cluster
    data, labels = make_blobs(num_samples, n_features=dimensions, centers=cluster_centers, cluster_std=dimensions*cluster_std_factor)

    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)
    normalized_centers = scaler.transform(cluster_centers)

    return normalized_data, labels, normalized_centers