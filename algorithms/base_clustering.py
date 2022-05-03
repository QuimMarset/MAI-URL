from abc import ABC, abstractmethod
import numpy as np


class BaseClustering(ABC):

    def __init__(self, num_clusters, max_iterations, is_forgy_initialization):
        self.num_clusters = num_clusters
        self.max_iterations = max_iterations
        self.is_forgy_initialization = is_forgy_initialization

    def initialize_centers_forgy(self, data):
        indices = np.random.choice(data.shape[0], self.num_clusters, replace=False)
        return data[indices]

    def initialzie_centers_random_partition(self, data):
        random_assignments = np.random.randint(0, self.num_clusters, data.shape[0])
        centers = [np.mean(data[random_assignments == center_index], axis=0) for center_index in range(self.num_clusters)]
        return np.array(centers)
    
    def initialize_cluster_centers(self, data):
        if self.is_forgy_initialization:
            return self.initialize_centers_forgy(data)
        else:
            return self.initialzie_centers_random_partition(data)

    def recompute_center_location(self, data, membership, weights):
        # Equation 1 in the paper
        normalization_factors = np.expand_dims(membership @ weights, axis=0).T
        unnormalized_centers = np.expand_dims(membership * weights, axis=1) @ data
        unnormalized_centers = np.squeeze(unnormalized_centers, axis=1)
        centers = unnormalized_centers / normalization_factors
        return centers

    @abstractmethod
    def fit(self, data):
        pass

    @abstractmethod
    def predict(self, data):
        pass