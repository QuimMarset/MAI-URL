from abc import ABC, abstractmethod
import numpy as np


class BaseClustering(ABC):

    def __init__(self, num_clusters, max_iterations, repetitions, is_forgy_initialization):
        self.num_clusters = num_clusters
        self.max_iterations = max_iterations
        self.repetitions = repetitions
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

    @abstractmethod
    def compute_cluster_membership(self, distances):
        pass

    @abstractmethod
    def compute_data_weights(self, distances):
        pass

    @abstractmethod
    def centers_have_changed(self, centers, old_centers):
        pass

    @abstractmethod
    def compute_objective_function(self, distances):
        pass

    def recompute_center_location(self, data, membership, weights):
        # Equation 1 in the paper
        normalization_factors = np.expand_dims(membership @ weights, axis=0).T
        unnormalized_centers = np.expand_dims(membership * weights, axis=1) @ data
        unnormalized_centers = np.squeeze(unnormalized_centers, axis=1)
        centers = unnormalized_centers / normalization_factors
        return centers

    def fit(self, data):
        centers = self.initialize_cluster_centers(data)
        iteration = 0
        centers_change = True

        while iteration < self.max_iterations and centers_change:
            old_centers = centers.copy()
            
            distances = np.linalg.norm(data - np.expand_dims(centers, axis=1), axis=2)**2
            membership = self.compute_cluster_membership(distances)
            weights = self.compute_data_weights(distances)
            centers = self.recompute_center_location(data, membership, weights)
            
            iteration += 1
            centers_change = self.centers_have_changed(centers, old_centers)
        
        objective_function = self.compute_objective_function(distances)
        return membership, centers, objective_function

    def predict(self, data):
        return self.compute_cluster_membership(data)