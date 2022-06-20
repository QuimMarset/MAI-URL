from algorithms.kmeans import KMeans
from algorithms.k_harmonic_means import KHarmonicMeans
from utils.metrics import *
from utils.plot_utils import *
import time


def run_kmeans(data, num_clusters, initial_centers, iterations, repetitions):
    kmeans = KMeans(num_clusters, iterations, repetitions)
    centers, membership, quality_values = kmeans.fit(data, initial_centers=initial_centers)
    return centers, membership, quality_values[-1]


def run_kharmonic(data, num_clusters, initial_centers, iterations, repetitions, p, convergence_threshold):
    kharmonic = KHarmonicMeans(num_clusters, iterations, p, convergence_threshold, repetitions)
    centers, membership, quality_values = kharmonic.fit(data, initial_centers=initial_centers)
    return centers, membership, quality_values[-1]


def run_semantic_segmentation(image, data, scaler, num_clusters, initial_centers, init_name, algorithm_name, image_name, 
                            save_path, algorithm_function, algorithm_params):

    start_time = time.time()
    centers, membership, clustering_quality = algorithm_function(data, num_clusters, initial_centers, **algorithm_params)
    end_time = time.time()

    restored_centers = scaler.inverse_transform(centers)
    centers_color_data = np.clip(np.uint8(restored_centers[:, 2:]), 0, 255)

    clustering = np.argmax(membership, axis=1)

    print(f'Semantic segmentaton of {image_name} using {algorithm_name} with {init_name} initialization: quality: {clustering_quality:.2f},')
    plot_image_segmentation(centers_color_data, clustering, image.shape, algorithm_name, init_name, image_name, save_path)
    print(f'Time to run the algorithm: {end_time - start_time:.2f}\n')