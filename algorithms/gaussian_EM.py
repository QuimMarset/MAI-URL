import numpy as np
from scipy.stats import multivariate_normal
from algorithms.base_clustering import BaseClustering


class GaussianEM(BaseClustering):

    def __init__(self, num_clusters, max_iterations, data, threshold):
        super().__init__(num_clusters, max_iterations)
        self.init_gaussian_parameters(data)
        self.threshold = threshold
        self.prev_log_likelihood = -np.inf

    def init_gaussian_parameters(self, data):
        num_dimensions = data.shape[1]
        # Regularization factor to avoid singular covariance matrices
        self.reg_cov = 1e-6*np.identity(num_dimensions)
        # Mus for each gaussian
        self.mus = np.random.randint(min(data[:, 0]), max(data[:, 0]), size=(self.num_clusters, num_dimensions))
        # Symmetric covariance matrices with 1 in the diagonals
        self.cov_matrices = np.zeros((self.num_clusters, num_dimensions, num_dimensions))
        for index in self.num_clusters:
            np.fill_diagonal(self.cov_matrices[index], 1)
        # Cluster prior (i.e. probability of a random point to belong to a cluster)
        self.pis = np.ones(self.num_clusters) / self.number_of_sources

    def perform_expectation_step(self, data):
        num_points = data.shape[0]
        posterior_data_cluster = np.zeros((self.num_clusters, num_points))

        gaussians = [multivariate_normal(mu, cov_matrix) for (mu, cov_matrix) in zip(self.mus, self.cov_matrices+self.reg_cov)]
        pdfs = np.array([gaussian.pdf(data) for gaussian in gaussians])

        for cluster_index, pi, in zip(range(self.num_clusters), self.pis):
            pdf = pdfs[cluster_index]
            numerator = pi * pdf
            denominator = np.sum(self.pis * pdfs, axis=0)
            posterior_data_cluster[cluster_index, :] = numerator / denominator

        return posterior_data_cluster

    def perform_maxmization_step(self, data, posterior_data_cluster):
        fraction_points = np.sum(posterior_data_cluster, axis=0)
        self.pis = fraction_points / np.sum(posterior_data_cluster)
        self.mus = np.sum(posterior_data_cluster * data, axis=0) / fraction_points
        self.cov_matrices = np.dot(posterior_data_cluster * (data - self.mus).T, (data - self.mus)) / fraction_points

    def compute_cluster_membership(self, distances):
        num_points = distances.shape[1]
        membership = np.zeros((self.num_clusters, num_points))
        membership[np.arange(self.num_clusters), np.argmin(distances, axis=0)] = 1
        return membership

    def compute_data_weights(self, data):
        posterior_data_cluster = self.perform_expectation_step(data)
        self.perform_maxmization_step(data, posterior_data_cluster)
        return posterior_data_cluster

    def has_converged(self, centers, old_centers):
        change = abs(self.log_likelihood - self.prev_log_likelihood)
        return change < self.threshold

    def compute_objective_function(self, data):
        gaussians = [multivariate_normal(mu, cov_matrix) for (mu, cov_matrix) in zip(self.mus, self.cov_matrices)]
        pdfs = np.array([gaussian.pdf(data) for gaussian in gaussians])
        self.log_likelihood = np.sum(np.log(np.sum([pi * pdf for (pi, pdf) in zip(self.pis, pdfs)], axis=0)))
        return self.log_likelihood    