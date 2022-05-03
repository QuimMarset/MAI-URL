import numpy as np


def compute_quality_metric(data, cluster_centers):
    # Squared root of the K-Means objective function
    distances = np.linalg.norm(data - np.expand_dims(cluster_centers, axis=1), axis=2)**2
    sum_squared_error = np.sum(np.min(distances, axis=0))
    return np.sqrt(sum_squared_error)
