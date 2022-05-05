import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import cycle
import matplotlib.colors as plt_colors
import numpy as np
import os
import seaborn as sns
from utils.name_utils import *


def get_cluster_colors(num_clusters):
    color_names = list(plt_colors.cnames.keys())
    color_names.remove('black')
    if num_clusters < len(color_names):
        return cycle(color_names)
    else:
        return cm.rainbow(np.linspace(0, 1, num_clusters))


def plot_2d_clustering(data, cluster_centers, assignment, colors):
    num_clusters = cluster_centers.shape[0]
    for (index, color) in zip(range(num_clusters), colors):
        cluster_data = data[assignment == index]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], color=color)
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], color='black', marker='x')


def plot_3d_clustering(data, cluster_centers, assignment, colors):
    num_clusters = cluster_centers.shape[0]
    ax = plt.axes(projection='3d')
    for (index, color) in zip(range(num_clusters), colors):
        cluster_data = data[assignment == index]
        ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2], color=color)
    ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2], color='black', marker='x')


def plot_clustering(data, cluster_centers, assignment, plot_title, fig_name, save_path):
    plt.figure(figsize=(8, 8))
    num_clusters = cluster_centers.shape[0]
    num_dimensions = data.shape[1]
    colors = get_cluster_colors(num_clusters)
    
    if num_dimensions < 3:
        plot_2d_clustering(data, cluster_centers, assignment, colors)
    else:
        plot_3d_clustering(data, cluster_centers, assignment, colors)

    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], color='black', marker='x')
    plt.title(plot_title)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{fig_name}.jpg'))
    plt.close()


def plot_cluster_centers(data, cluster_centers, algorithm_name, dataset_name, initialization_name, quality, clusters_found, iterations, save_path):
    plt.figure(figsize=(12, 8))
    plt.scatter(data[:, 0], data[:, 1], color='b', label='Original Data')
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], color='r', label='Cluster Centers')
    title = get_plot_cluster_centers_title(algorithm_name, dataset_name, initialization_name, quality, clusters_found, iterations)
    fig_name = get_plot_cluster_centers_fig_name(algorithm_name, dataset_name, initialization_name, iterations)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
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



def plot_k_optimization(dataset_name, k_values, metric_values, PCA_f, path):
    plt.figure()
    plt.plot(k_values, metric_values)
    plt.xlabel('K')
    plt.xticks(range(k_values[0], k_values[-1]+1))
    plt.ylabel('Silhouette score')
    
    title = f'K-Means++ K optimization on {dataset_name} dataset'
    file_name = f'{path}k-means++_k_optimization_{dataset_name}'
    
    if PCA_f:
        title += " using PCA"
        file_name += "_PCA"
    
    plt.title(title)
    plt.savefig(file_name + ".png")
    plt.close()


def plot_pca_explained_variance(explained_variance_ratios, dataset_name, save_path):
    cumulative = np.cumsum(explained_variance_ratios)
    components = range(len(explained_variance_ratios))

    comp_90 = [index for index in range(len(cumulative)) if cumulative[index] >= 0.9][0]
    ratio_90 = cumulative[comp_90]

    plt.bar(components, explained_variance_ratios, alpha=0.5, align='center', label='Individual explained variance ratio')
    plt.step(components, cumulative, where='mid', label='Cumulative explained variance ratio')
    plt.scatter(comp_90, ratio_90, c = 'r', label='PCs reaching 90% of explained variance', s=10)
    plt.annotate(f'{comp_90+1} PCs - {ratio_90:.3f}', (comp_90, ratio_90), xytext=(10, -20), textcoords='offset points')
    plt.ylabel('Explained variance ratio by PC')
    plt.xlabel('PC index')
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.legend()
    plt.title(f'PCA PCs explained variance on {dataset_name} dataset')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'explained_variance_{dataset_name}.jpg'))
    plt.close()