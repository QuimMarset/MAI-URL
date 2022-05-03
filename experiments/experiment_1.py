import numpy as np
from algorithms.kmeans import KMeans
from algorithms.fuzzy_c_means import FuzzyCMeans
from algorithms.gaussian_EM import GaussianEM
from algorithms.k_harmonic_means import KHarmonicMeans
from algorithms.hybrid import Hybrid1, Hybrid2
from experiments.metrics import compute_quality_metric
from datasets.birch_data import generate_birch_data, plot_data


def run_k_means(num_clusters, iterations, repetitions, initial_centers, data):
    k_means = KMeans(num_clusters, iterations, repetitions, initial_centers)
    clustering, cluster_centers, iterations_metric = k_means.fit(data)
    plot_clustering(data, clustering, cluster_centers, iterations_metric[-1])
    return iterations_metric


def run_fuzzy_c_means(num_clusters, iterations, fuzzy_degree, threshold, repetitions, data, initial_centers):
    fuzzy_c_means = FuzzyCMeans(num_clusters, iterations, fuzzy_degree, threshold, repetitions, initial_centers)
    clustering, cluster_centers, iterations_metric = fuzzy_c_means.fit(data)
    plot_clustering(data, clustering, cluster_centers, iterations_metric[-1])
    return iterations_metric


def run_k_harmonic_means(num_clusters, iterations, p, threshold, repetitions, data, initial_centers):
    k_harmonic_means = KHarmonicMeans(num_clusters, iterations, p, threshold, repetitions, initial_centers)
    clustering, cluster_centers, iterations_metric = k_harmonic_means.fit(data)
    plot_clustering(data, clustering, cluster_centers, iterations_metric[-1])
    return iterations_metric


def run_gaussian_EM(num_clusters, iterations, threshold, cov_init_value, data, initial_centers):
    gaussian_em = GaussianEM(num_clusters, iterations, threshold, repetitions, initial_centers)
    clustering, cluster_centers, iterations_metric = fuzzy_c_means.fit(data)
    plot_clustering(data, clustering, cluster_centers, iterations_metric[-1])
    return iterations_metric


def run_hybrid_1(num_clusters, iterations, p, threshold, data, initial_centers):
    pass


def run_hybrid_2(num_clusters, iterations, p, threshold, data, initial_centers):
    pass


def experiment_with_initial_centers(num_clusters, iterations, repetitions, birch_data, initial_centers, fuzzy_degree,
    fcm_threshold, gem_threshold, khm_threshold, p, cov_init_value):

    k_means_metric = run_k_means(num_clusters, iterations)
    fuzzy_c_means_metric = run_fuzzy_c_means()
    k_harmonic_means_metric = run_k_harmonic_means()
    gaussian_EM_metric = run_gaussian_EM()
    hybrid_1_metric = run_hybrid_1()
    hybrid_2_metric = run_hybrid_2()

    metrics = []
    plot_clustering_comparison_metrics(metrics)



def perform_experiment_1():

    birch_data, true_clustering, true_cluster_centers = generate_birch_data(grid_size=10, 
        start_coodinate=30, points_per_cluster=100)

    plot_clustering(birch_data, true_clustering, true_cluster_centers)

    forgy_initial_centers = initialize_centers_forgy()
    random_partition_initial_centers = initialize_centers_random_partition()
    experiment_with_initial_centers(birch_data, forgy_initial_centers)
    experiment_with_initial_centers(birch_data, random_partition_initial_centers)