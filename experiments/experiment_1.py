from algorithms.kmeans import KMeans
from algorithms.fuzzy_c_means import FuzzyCMeans
from algorithms.gaussian_EM import GaussianEM
from algorithms.k_harmonic_means import KHarmonicMeans
from algorithms.hybrid import Hybrid1, Hybrid2
from metrics import *
from cluster_initialization import *
from datasets.birch_data import generate_birch_data
from utils.plot_utils import *
from constants import *
import time


save_path = './results/'

grid_size = 10
num_clusters = grid_size * grid_size
points_per_cluster = 100
center_distance = 4 * np.sqrt(2)

iterations = 100
repetitions = 1

fuzzy_degree = 1.3
harmonic_p = 3.5
gem_cov_diagonal = 0.2

fcm_threshold = 0.001
harmonic_threshold = 0.001
gem_threshold = 0.001


k_means = KMeans(num_clusters, iterations, repetitions)
fuzzy_c_means = FuzzyCMeans(num_clusters, iterations, fuzzy_degree, fcm_threshold, repetitions)
k_harmonic_means = KHarmonicMeans(num_clusters, iterations, harmonic_p, harmonic_threshold, repetitions)
gaussian_EM = GaussianEM(num_clusters, iterations, gem_threshold, gem_cov_diagonal)
hybrid_1 = Hybrid1(num_clusters, iterations, harmonic_p, harmonic_threshold, repetitions)
hybrid_2 = Hybrid2(num_clusters, iterations, harmonic_p, harmonic_threshold, repetitions)

clustering_objects = [k_means, fuzzy_c_means, k_harmonic_means, hybrid_1, hybrid_2]
algorithm_names = [KMEANS, FCMEANS, KHARMONIC, HYBRID1, HYBRID2]


def run_algorithm(clustering_object, data, true_centers, cluster_radius, initial_centers, algorithm_name, init_name, save_path):
    _, cluster_centers, _ = clustering_object.fit(data, initial_centers)
    clustering_quality = compute_quality_metric(data, cluster_centers)
    num_true_found = compute_true_clusters_found(true_centers, cluster_radius, cluster_centers)
    plot_cluster_centers(data, cluster_centers, algorithm_name, BIRCH, init_name, clustering_quality, num_true_found, iterations, save_path)
    print(f'{algorithm_name} clustering on {BIRCH} data with {init_name} initialization: quality: {clustering_quality:.2f}, true clusters found: {num_true_found}')


def perform_experiment_forgy(data, true_centers, cluster_radius):
    initial_centers = initialize_centers_forgy(num_clusters, data)
    for clustering_object, algorithm_name in zip(clustering_objects, algorithm_names):
        run_algorithm(clustering_object, data, true_centers, cluster_radius, initial_centers, algorithm_name, FORGY, save_path)
        

def perform_experiment_random_partition(data, true_centers, cluster_radius):
    initial_centers = initialzie_centers_random_partition(num_clusters, data)
    for clustering_object, algorithm_name in zip(clustering_objects, algorithm_names):
        run_algorithm(clustering_object, data, true_centers, cluster_radius, initial_centers, algorithm_name, RANDPART, save_path)
        

def perform_experiment_1():
    start_time = time.time()

    birch_data, true_clustering, true_cluster_centers = generate_birch_data(grid_size, center_distance, points_per_cluster)

    dist = np.linalg.norm(true_cluster_centers[1] - true_cluster_centers[0])
    cluster_radius = dist / 4

    perform_experiment_forgy(birch_data, true_cluster_centers, cluster_radius)
    perform_experiment_random_partition(birch_data, true_cluster_centers, cluster_radius)

    print(f'Time to run experiment 1: {(time.time() - start_time):.2f} seconds')