from algorithms.kmeans import KMeans
from algorithms.fuzzy_c_means import FuzzyCMeans
from algorithms.gaussian_EM import GaussianEM
from algorithms.k_harmonic_means import KHarmonicMeans
from algorithms.hybrid import Hybrid1, Hybrid2
from metrics import *
from cluster_initialization import *
from datasets.pelleg_moore_data import generate_pelleg_moore_data
from utils.plot_utils import *
import time
from constants import *

save_path = './results/experiment_2/'

num_clusters = 50
samples_per_cluster = 50
cluster_std_factor = 0.012

iterations = 100
repetitions = 1

fuzzy_degree = 1.3
harmonic_p = 3.5
gem_cov_diagonal = 0.2
soft_cluster_threshold = 0.001


k_means = KMeans(num_clusters, iterations, repetitions)
fuzzy_c_means = FuzzyCMeans(num_clusters, iterations, fuzzy_degree, soft_cluster_threshold, repetitions)
k_harmonic_means = KHarmonicMeans(num_clusters, iterations, harmonic_p, soft_cluster_threshold, repetitions)
gaussian_EM = GaussianEM(num_clusters, iterations, soft_cluster_threshold, gem_cov_diagonal)
hybrid_1 = Hybrid1(num_clusters, iterations, harmonic_p, soft_cluster_threshold, repetitions)
hybrid_2 = Hybrid2(num_clusters, iterations, harmonic_p, soft_cluster_threshold, repetitions)

clustering_objects = [k_means, fuzzy_c_means, k_harmonic_means, hybrid_1, hybrid_2]
algorithm_names = [KMEANS, FCMEANS, KHARMONIC, HYBRID1, HYBRID2]
k_means_convergence = KMeans(num_clusters, 100000, repetitions)


def run_experiment_dataset_initialization(data, true_centers, initial_centers):
    optimal_quality_values = k_means_convergence.fit_experiment_2(data, initial_centers)
    ratios = []

    for clustering_object in clustering_objects:
        quality_values = clustering_object.fit_experiment_2(data, initial_centers)
        ratios.append(compute_quality_metric_ratio(optimal_quality_values, quality_values))

    return ratios, min(len(optimal_quality_values), len(quality_values))


def run_experiment_dataset(dimensions):
    data, _, true_centers = generate_pelleg_moore_data(dimensions, num_clusters, samples_per_cluster, cluster_std_factor)
    initial_centers_forgy = initialize_centers_forgy(num_clusters, data)
    initial_centers_random_part = initialzie_centers_random_partition(num_clusters, data)

    ratios_forgy, min_iterations_forgy = run_experiment_dataset_initialization(data, true_centers, initial_centers_forgy)
    ratios_rand_part, min_iterations_rand_part = run_experiment_dataset_initialization(data, true_centers, initial_centers_random_part)
    return ratios_forgy, ratios_rand_part, min_iterations_forgy, min_iterations_rand_part


def run_experiment_dimension(dimensions, num_datasets):
    ratios_forgy = np.zeros((num_datasets, len(clustering_objects), iterations))
    ratios_rand_part = np.zeros((num_datasets, len(clustering_objects), iterations))

    min_iterations_forgy = iterations
    min_iterations_rand_part = iterations

    for i in range(num_datasets):
        ratios_forgy_dataset, ratios_rand_part_dataset, min_iterations_forgy_i, min_iterations_rand_part_i = run_experiment_dataset(dimensions)
        ratios_forgy[i, :, :min_iterations_forgy_i] = ratios_forgy_dataset
        ratios_rand_part[i, :, :min_iterations_rand_part_i] = ratios_rand_part_dataset

        if min_iterations_forgy_i < min_iterations_forgy:
            min_iterations_forgy = min_iterations_forgy_i
        if min_iterations_rand_part_i < min_iterations_rand_part:
            min_iterations_rand_part = min_iterations_rand_part_i

    return ratios_forgy, ratios_rand_part, min_iterations_forgy, min_iterations_rand_part


def perform_experiment_2():
    start_time = time.time()

    dimensions_list = [2, 4, 6]
    num_datasets = 100

    for dimensions in dimensions_list:

        ratios_forgy_dimension, ratios_rand_part_dimension, min_iter_forgy, min_iter_rand_part = run_experiment_dimension(dimensions, num_datasets)

        avg_ratios_forgy = np.mean(ratios_forgy_dimension[:, :, :min_iter_forgy], axis=0)
        plot_algorithms_ratio_quality_comparison(avg_ratios_forgy, algorithm_names, PELLEG, FORGY, dimensions, num_datasets, save_path)

        avg_ratios_rand = np.mean(ratios_rand_part_dimension[:, :, :min_iter_forgy], axis=0)
        plot_algorithms_ratio_quality_comparison(avg_ratios_rand, algorithm_names, PELLEG, FORGY, dimensions, num_datasets, save_path)

    print(f'Time to run experiment 2: {time.time() - start_time:.2f} seconds')