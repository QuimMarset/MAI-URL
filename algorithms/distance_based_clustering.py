from algorithms.base_clustering import BaseClustering
from abc import abstractmethod
from scipy.spatial.distance import cdist
from metrics import compute_quality_metric


class DistanceBasedClustering(BaseClustering):

    def __init__(self, num_clusters, max_iterations):
        super().__init__(num_clusters, max_iterations)


    @abstractmethod
    def compute_cluster_membership(self, distances):
        pass


    @abstractmethod
    def compute_data_weights(self, distances):
        pass


    @abstractmethod
    def has_converged(self, centers, old_centers):
        pass


    @abstractmethod
    def compute_objective_function(self, distances):
        pass


    def fit(self, data, initial_centers):
        self.centers = initial_centers
        iteration = 0
        has_converged = False

        while iteration < self.max_iterations and not has_converged:
            old_centers = self.centers.copy()
            
            distances = cdist(data, self.centers)
            membership = self.compute_cluster_membership(distances)
            weights = self.compute_data_weights(distances)
            self.centers = self.recompute_center_location(data, membership, weights)
            
            iteration += 1
            has_converged = self.has_converged(self.centers, old_centers)
        
        objective_function = self.compute_objective_function(distances)
        distances = cdist(data, self.centers)
        membership = self.compute_cluster_membership(distances)
        return membership, self.centers, objective_function


    def predict(self, data):
        distances = cdist(data, self.centers)
        return self.compute_cluster_membership(distances)

    
    def fit_experiment_2(self, data, initial_centers):
        self.centers = initial_centers
        iteration = 0
        has_converged = False

        kmeans_quality_values = []

        while iteration < self.max_iterations and not has_converged:
            old_centers = self.centers.copy()
            
            distances = cdist(data, self.centers)
            membership = self.compute_cluster_membership(distances)
            weights = self.compute_data_weights(distances)
            self.centers = self.recompute_center_location(data, membership, weights)
            
            iteration += 1
            has_converged = self.has_converged(self.centers, old_centers)

            kmeans_quality_values.append(compute_quality_metric(data, self.centers))
        
        return kmeans_quality_values