import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import cycle
import matplotlib.colors as plt_colors
import numpy as np
import os
import seaborn as sns
from utils.string_utils import *


def get_cluster_colors(num_clusters):
    color_names = list(plt_colors.cnames.keys())
    color_names.remove('black')
    if num_clusters < len(color_names):
        return cycle(color_names)
    else:
        return cm.rainbow(np.linspace(0, 1, num_clusters))


def plot_generated_synthetic_data(data, true_centers, true_clustering, dataset_name, save_path, points_per_cluster=None):
    plt.figure(figsize=(10, 8))
    num_clusters = true_centers.shape[0]
    colors = get_cluster_colors(num_clusters)
    
    for (index, color) in zip(range(num_clusters), colors):
        cluster_data = data[true_clustering == index]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], color=color)
    plt.scatter(true_centers[:, 0], true_centers[:, 1], color='black', marker='x', label='Cluster centers')

    title = f'{dataset_name} synthetic data with {num_clusters} clusters'
    if points_per_cluster:
        title += f' and {points_per_cluster} samples per cluster'

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    dataset_file_name = dataset_names_in_files[dataset_name]
    plt.savefig(os.path.join(save_path, f'{dataset_file_name}_generated_data.jpg'))
    plt.close()


def plot_initial_cluster_centers(data, initial_centers, dataset_name, init_name, save_path):
    plt.figure(figsize=(10, 8))
    plt.scatter(data[:, 0], data[:, 1], color='b', label='Original data')
    plt.scatter(initial_centers[:, 0], initial_centers[:, 1], color='r', label='Initial cluster centers')
    data_name_file = dataset_names_in_files[dataset_name]
    init_name_file = initialization_names_in_files[init_name]
    plt.title(f'{init_name} initial centers on {dataset_name} data')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'initial_centers_{data_name_file}_{init_name_file}.jpg'))
    plt.close()


def plot_predicted_clustering(data, centers, clustering, dataset_name, algorithm_name, init_name, quality, save_path):
    plt.figure(figsize=(10, 8))
    num_clusters = centers.shape[0]
    colors = get_cluster_colors(num_clusters)
    
    for (index, color) in zip(range(num_clusters), colors):
        cluster_data = data[clustering == index]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], color=color)
    plt.scatter(centers[:, 0], centers[:, 1], color='black', marker='x', label='Predicted centers')

    plt.legend()
    alg_name_file = algorithm_names_in_files[algorithm_name]
    data_name_file = dataset_names_in_files[dataset_name]
    init_name_file = initialization_names_in_files[init_name]
    plt.title(f'{algorithm_name} on {dataset_name} data: {init_name} initialization, quality {quality:.2f}')
    plt.tight_layout()
    fig_name = f'clustering_{alg_name_file}_{data_name_file}_{init_name_file}'
    plt.savefig(os.path.join(save_path, f'{fig_name}.jpg'))
    plt.close()


def plot_algorithms_ratio_quality_comparison(ratios, algorithm_names, dataset_name, init_name, dimensions, num_datasets, save_path):
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))

    for algorithm_values, algorithm_name in zip(ratios, algorithm_names):
        plt.plot(algorithm_values, label=algorithm_name, marker='o')

    plt.xlabel('Iteration')
    plt.ylabel('Average k-means quality ratio')
    plt.title(f'Algorithms convergence using {init_name} initialization averaging over {num_datasets} {dataset_name} datasets with {dimensions} dimensions')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'cluster_ratio_comparison_{init_name}_{dataset_name}_{dimensions}_{num_datasets}.jpg'))
    plt.close()


def plot_k_optimization(dataset_name, k_values, results, algorithm_name, save_path):
    sns.set(style="whitegrid")
    _, axs = plt.subplots(1, 3, figsize=(15, 6))

    for (ax, metric_name) in zip(axs, results):
        ax.plot(k_values, results[metric_name])
        ax.set_title(metric_name)
        ax.set_ylabel(f'{metric_name} score')
        ax.set_xlabel('Number of clusters')
        ax.set_xticks(k_values)

    plt.suptitle(f'Number of clusters optimization of {algorithm_name} on {dataset_name} dataset')
    plt.tight_layout()
    algorithm_file_name = algorithm_names_in_files[algorithm_name]
    dataset_file_name = dataset_names_in_files[dataset_name]
    plt.savefig(os.path.join(save_path, f'{algorithm_file_name}_k_optimization_{dataset_file_name}.jpg'), dpi=300)
    plt.close()


def plot_pca_explained_variance(explained_variance_ratios, dataset_name, save_path):
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    components = range(len(explained_variance_ratios))
    cumulative = np.cumsum(explained_variance_ratios)

    comp_90 = [index for index in range(len(cumulative)) if cumulative[index] >= 0.9][0]
    ratio_90 = cumulative[comp_90]

    plt.bar(components, cumulative, align='center', alpha=0.5, label='Cumulative explained variance ratio')
    plt.scatter(comp_90, ratio_90, c = 'r', label='PCs reaching 90% of explained variance', s=50)
    plt.annotate(f'{comp_90+1} PCs - {ratio_90:.3f}', (comp_90, ratio_90), xytext=(-20, 10), textcoords='offset points')
    
    plt.xlabel('Principal Components')
    plt.ylabel('Explained variance ratio')
    plt.legend()
    plt.title(f'PCA cumulative explained variance ratio on {dataset_name} dataset')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'explained_variance_{dataset_names_in_files[dataset_name]}.jpg'))
    plt.close()


def write_cluster_centers_to_txt(centers, feature_names, algorithm_name, dataset_name, save_path):
    path = os.path.join(save_path, 
        f'cluster_centers_{algorithm_names_in_files[algorithm_name]}_{dataset_names_in_files[dataset_name]}.txt')

    with open(path, 'w') as file:
        for (i, center) in enumerate(centers):
            file.write(f'Center {i}:\n')
            for j in range(centers.shape[1]):
                file.write(f'  {feature_names[j]}: {center[j]}\n')
            file.write('\n')


def plot_clustering_3d(data, pred_assignment, feature_names, dataset_name, algorithm_name, save_path):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    cluster_indices = np.unique(pred_assignment)
    num_clusters = len(cluster_indices)
    algorithm_file_name = algorithm_names_in_files[algorithm_name]
    dataset_file_name = dataset_names_in_files[dataset_name]

    for cluster_index in cluster_indices:
        cluster_data = data[pred_assignment == cluster_index]
        ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2], label=f'Cluster {cluster_index+1}', s=10)

    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_zlabel(feature_names[2])
    plt.legend()
    plt.title(f'{algorithm_name} clustering with {num_clusters} clusters on {dataset_name} dataset')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'clustering_{algorithm_file_name}_{num_clusters}_clusters_{dataset_file_name}.jpg'), dpi=300)
    plt.close()


def plot_image_segmentation(centers, clustering, image_shape, algorithm_name, init_name, image_name, save_path):
    segmented_image = segmented_image = centers[clustering].reshape(image_shape)
    plt.figure(figsize=(8, 8))
    plt.imshow(segmented_image)
    plt.axis('off')
    algorithm_file_name = algorithm_names_in_files[algorithm_name]
    init_file_name = initialization_names_in_files[init_name]
    plt.savefig(os.path.join(save_path, f'segmented_{image_name}_{algorithm_file_name}_{init_file_name}.jpg'), bbox_inches='tight')
    plt.close()