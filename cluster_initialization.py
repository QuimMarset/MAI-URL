import numpy as np


def initialize_centers_forgy(num_clusters, data):
    indices = np.random.choice(data.shape[0], num_clusters, replace=False)
    return data[indices]


def initialzie_centers_random_partition(num_clusters, data):
    random_assignments = np.random.randint(0, num_clusters, data.shape[0])
    centers = [np.mean(data[random_assignments == center_index], axis=0) for center_index in range(num_clusters)]
    return np.array(centers)

def initialize_kmeans_plus_plus_centers(num_clusters, data):
    pass