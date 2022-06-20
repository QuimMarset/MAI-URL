import numpy as np
from scipy.spatial.distance import cdist


def initialize_centers_forgy(num_clusters, data):
    indices = np.random.choice(data.shape[0], num_clusters, replace=False)
    return data[indices]


def initialzie_centers_random_partition(num_clusters, data):
    random_assignments = np.random.randint(0, num_clusters, data.shape[0])
    centers = [np.mean(data[random_assignments == center_index], axis=0) for center_index in range(num_clusters)]
    return np.array(centers)


def initialize_kmeans_plus_plus_centers(num_clusters, data):
    centers = []
    # Initialize the first centroid picking a random instance from the data
    random_index = np.random.choice(data.shape[0])
    centers.append(data[random_index])

    i = 1
    while i < num_clusters:
        # Compute the distances of each instance to each computed center
        distances = cdist(data, centers)
        # Pick for each instance the minimum to one of the possible centers
        min_distances_to_center = np.min(distances, axis=1)**2
        # Compute a probability distribution using the computed distances
        probabilities = min_distances_to_center / np.sum(min_distances_to_center)
        # Select a new center from the data using the probability distribution
        random_index = np.random.choice(data.shape[0], p=probabilities)
        centers.append(data[random_index])
        i += 1
    
    return np.array(centers)