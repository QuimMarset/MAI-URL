import numpy as np
from scipy.spatial.distance import cdist
from sklearn import metrics
import os
from utils.string_utils import *


def compute_quality_metric(data, cluster_centers):
    # Squared root of the K-Means objective function
    distances = cdist(data, cluster_centers)**2
    sum_squared_error = np.sum(np.min(distances, axis=1))
    return sum_squared_error


def compute_quality_metric_ratio(optimal_quality_values, predicted_quality_values, iterations):
    num_iterations_optimal = len(optimal_quality_values)
    num_iterations_predicted = len(predicted_quality_values)

    optimal_qualities = np.zeros(iterations)
    optimal_qualities[:num_iterations_optimal] = optimal_quality_values
    optimal_qualities[num_iterations_optimal:] = optimal_quality_values[-1]

    predicted_qualities = np.zeros(iterations)
    predicted_qualities[:num_iterations_predicted] = predicted_quality_values
    predicted_qualities[num_iterations_predicted:] = predicted_quality_values[-1]

    return np.sqrt(predicted_qualities / optimal_qualities)


""" 
Internal Validation
    Davies Bouldin coefficient
    Calisnki-Harabasz coefficient
    Silhouette coefficient (problema alt cost computacional)
"""

def compute_internal_indices(data, clusters):
    # The lower value is 0, and we want to minimize this score
    davies_bouldin_score = metrics.davies_bouldin_score(data, clusters)
    # Ranged between -1 and 1, and we want to maximize this score
    silhouette_coefficient = metrics.silhouette_score(data, clusters)
    # We want to maximize this score
    calinski_harabasz_score = metrics.calinski_harabasz_score(data, clusters)
    
    metrics_dict = {'Davies&Bouldin Score' : davies_bouldin_score, 'Silhouette Coefficient' : silhouette_coefficient,
        'Calinski-Harabasz Score' : calinski_harabasz_score}
    return metrics_dict


"""
External Validation
    Purity
    Contingency Matrix -> Accuracy, Precision, Recall, F-Measure
    Adjusted Rand Index
    Fowlkes-Mallows Score
    Adjusted Mutual Information
    Completeness
"""

def compute_purity(clustering, true_clustering):
    contingency_matrix = metrics.cluster.contingency_matrix(true_clustering, clustering)
    purity = np.sum(np.max(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
    return purity


def compute_pair_confusion_matrix_metrics(clustering, true_clustering):
    pair_conf_matrix = metrics.cluster.pair_confusion_matrix(true_clustering, clustering)
    true_positives = pair_conf_matrix[1, 1]
    true_negatives = pair_conf_matrix[0, 0]
    false_positives = pair_conf_matrix[0, 1]
    false_negatives = pair_conf_matrix[1, 0]

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = (2 * precision * recall) / (precision + recall)
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)

    return accuracy, precision, recall, f1_score
    

def compute_external_indices(data, clustering, true_clustering):
    (accuracy, precision, recall, f1_score) = compute_pair_confusion_matrix_metrics(clustering, true_clustering)

    # The homogeneity of data in the different clusters. We want to maximize. 
    # Does not work well with cluster imbalance and does not penalize having a lot of clusters
    purity = compute_purity(clustering, true_clustering)

    # Adjusted version of the Rand Index (which can be seen as an accuracy in a binary classification problem). 
    # It ranges from -1 to 1, being 1 a perfect match
    adjusted_rand_index = metrics.adjusted_rand_score(true_clustering, clustering)
    
    # Geometric mean between precision and recall. Ranges from 0 to 1, being 1 a good similarity between clusters
    fowlkes_mallows_score = metrics.fowlkes_mallows_score(true_clustering, clustering)
    
    # Adjusted version of the Mutual Information (giving a measure of reduction of uncertainty of the obtained 
    # clusters if we know the real ones) to account for the higher values whenre there is a big number of clusters. 
    # Ranges from 0 to 1 being 1 a perfect match.
    adjusted_mutual_information_score = metrics.adjusted_mutual_info_score(true_clustering, clustering)
    
    # Ranges from 0 to 1, being 1 when the data of a cluster belongs to the same class
    homogeneity_score = metrics.homogeneity_score(true_clustering, clustering)

    # Ranges from 0 to 1 being 1 when the data of a given class are members of the same cluster
    completeness_score = metrics.completeness_score(true_clustering, clustering)

    # Ranges from 0 to 1, harmonic mean of completeness and homogeneity
    v_measure = metrics.v_measure_score(true_clustering, clustering)

    metrics_dict = { 
        'Confusion Matrix' : [accuracy, precision, recall, f1_score],
        'Purity' :  purity,
        'AdjustedRand Index' : adjusted_rand_index,
        'Fowlkes Mallows' : fowlkes_mallows_score,
        'Adjusted Mutual Information' : adjusted_mutual_information_score,
        'Completeness Score' : completeness_score,
        'Homogeneity Score' : homogeneity_score,
        'V-Measure Score' : v_measure
    }
    return metrics_dict


def evaluate_resulting_clusters(data, true_clustering, predicted_clustering, dataset_name, algorithm_name, save_path):

    internal_results = compute_internal_indices(data, predicted_clustering)
    external_results = compute_external_indices(data, predicted_clustering, true_clustering)

    algorithm_file_name = algorithm_names_in_files[algorithm_name]
    dataset_file_name = dataset_names_in_files[dataset_name]

    file_path = os.path.join(save_path, f'{algorithm_file_name}_eval_metrics_{dataset_file_name}.txt')
    
    with open(file_path, 'w') as file:

        file.write('Internal Metrics:\n\n')
        for key in internal_results:
            value = internal_results.get(key)
            file.write(f'{key}: {value:.2f}\n')

        file.write('\nExternal Metrics:\n\n')
        for key in external_results:
            value = external_results.get(key)
            if key == 'Confusion Matrix':
                file.write(f'{key}(accuracy): {value[0]:.2f}\n')
                file.write(f'{key}(precision): {value[1]:.2f}\n')
                file.write(f'{key}(recall): {value[2]:.2f}\n')
                file.write(f'{key}(f1-score): {value[3]:.2f}\n')
            else:
                file.write(f'{key}: {value:.2f}\n')