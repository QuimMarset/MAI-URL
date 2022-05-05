from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from algorithms.k_harmonic_means import KHarmonicMeans
from datasets.clean_adult import preprocess_adult
from utils.plot_utils import plot_k_optimization, plot_pca_explained_variance, plot_clustering
from metrics import evaluate_resulting_clusters
from constants import ADULT, KHARMONIC
from cluster_initialization import *


save_path = './results/experiment_4'

optimal_adult_pcs = 23
min_k = 2
max_k = 15

iterations = 500
repetitions = 15
p = 3.5
threshold = 0.001


optimal_k_sklearn_kmeans = 4
optimal_k_kharmonic = 4



def run_sklearn_kmeans(data, num_clusters, iterations, repetitions):
    kmeans = KMeans(num_clusters, n_init=repetitions, max_iter=iterations)
    clustering = kmeans.fit_predict(data)
    return clustering


def run_kharmonic(data, num_clusters, p, threshold, iterations, repetitions):
    initial_centers = initialize_kmeans_plus_plus_centers(num_clusters, data)
    kharmonic = KHarmonicMeans(num_clusters, iterations, p, threshold, repetitions)
    clustering, _, _ = kharmonic.fit(data, initial_centers)
    return clustering


def search_optimal_number_of_clusters(data, min_k, max_k, save_path):
    k_values = list(range(min_k, max_k+1))
    num_repetitions = 3

    metric_values_kmeans = []
    metric_values_kharmonic = []

    for k in k_values:
        print(f'\tk: {k}/{max_k}')

        average_silhouette_kmeans = 0
        average_silhouette_kharmonic = 0

        for _ in range(num_repetitions):

            kmeans_clustering = run_sklearn_kmeans()
            kharmonic_clustering = run_kharmonic()

            average_silhouette_kmeans += silhouette_score(data, kmeans_clustering)
            average_silhouette_kharmonic += silhouette_score(data, kharmonic_clustering)

        metric_values_kmeans.append(average_silhouette_kmeans/num_repetitions)
        metric_values_kharmonic.append(average_silhouette_kharmonic/num_repetitions)

    plot_k_optimization(ADULT, k_values, metric_values_kmeans, 'Sklearn-K-Means', save_path)
    plot_k_optimization(ADULT, k_values, metric_values_kharmonic, KHARMONIC, save_path)


def run_algorithms_with_optimal_k(data, true_clustering, save_path):
    kmeans_clustering = run_sklearn_kmeans(data, optimal_k_sklearn_kmeans, iterations, repetitions)
    evaluate_resulting_clusters(kmeans_clustering, true_clustering)
    plot_clustering(data, kmeans_clustering, save_path)

    kharmonic_clustering = run_kharmonic(data, optimal_k_kharmonic, p, threshold, iterations, repetitions)
    evaluate_resulting_clusters(kharmonic_clustering, true_clustering)
    plot_clustering(data, kharmonic_clustering, save_path)


def perform_experiment_4():
    adult_data, true_clustering = preprocess_adult('./datasets/adult.arff')
    
    pca = PCA()
    pca.fit(adult_data)
    plot_pca_explained_variance(pca.explained_variance_ratio_, ADULT, save_path)

    pca = PCA(optimal_adult_pcs)
    reduced_data = pca.fit_transform(adult_data)

    search_optimal_number_of_clusters(adult_data, min_k, max_k, save_path)

    run_algorithms_with_optimal_k(adult_data, true_clustering, save_path)