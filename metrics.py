import numpy as np
from scipy.spatial.distance import cdist
from sklearn import metrics


def compute_quality_metric(data, cluster_centers):
    # Squared root of the K-Means objective function
    distances = cdist(data, cluster_centers)
    sum_squared_error = np.sum(np.min(distances, axis=1))
    return np.sqrt(sum_squared_error)


def is_center_inside_true_cluster(true_center, cluster_radius, predicted_center):
    dist = np.linalg.norm(true_center - predicted_center)
    return dist <= cluster_radius


def compute_true_clusters_found(true_centers, cluster_radius, predicted_centers):
    num_true_found = 0
    for true_center in true_centers:
        for predicted_center in predicted_centers:
            if is_center_inside_true_cluster(true_center, cluster_radius, predicted_center):
                num_true_found += 1
                break
    return num_true_found
    

def compute_quality_metric_ratio(optimal_quality_values, predicted_quality_values):
    ratios = []
    for quality_optimal, quality_predicted in zip(optimal_quality_values, predicted_quality_values):
        ratios.append(np.sqrt(quality_predicted / quality_optimal))
    return ratios


""" 
Internal Validation

    Davies Bouldin coefficient
    Calisnki-Harabasz coefficient
    Silhouette coefficient (problema alt cost computacional)
"""

def compute_internal_indices(data, clusters):
    """
    Compute multiple Internal Indices to unsupervisedly evaluate the resulting clusters
    :param data: The dataset containing the instances to clusterize
    :param clusters: The cluster assignation to each instance
    :return: A dictionary containing multiple Internal Index to evaluate the clusters
    """
    # The lower value is 0, and we want to minimize this score
    davies_bouldin_score = metrics.davies_bouldin_score(data, clusters)
    # Ranged between -1 and 1, and we want to maximize this score
    silhouette_coefficient = metrics.silhouette_score(data, clusters)
    #We want to maximize this score
    calinski_harabasz_score = metrics.calinski_harabasz_score(data, clusters)
    
    metrics_dict = {'Davies&Bouldin Score' : davies_bouldin_score, 'Silhouette Coefficient' : silhouette_coefficient,
        'Calinski-Harabasz Score' : calinski_harabasz_score}
    return metrics_dict

"""
External Validation

    Contingency Matrix -> Accuracy, Precision, Recall, F-Measure
    Adjusted Rand Index
    Fowlkes-Mallows Score
    Adjusted Mutual Information
    Homogeneity, Completeness and V-Measure
"""

def compute_purity(clusters, true_labels):
    contingency_matrix = metrics.cluster.contingency_matrix(true_labels, clusters)
    purity = np.sum(np.max(contingency_matrix, axis=0))/np.sum(contingency_matrix)
    return purity


def compute_precision_recall_accuracy(clusters, true_labels):
    """
    Computes the Contingency Matrix (i.e. considering pairs of instances and determining if each instance of the pair 
    is in the same cluster and/or class) to extract measures of Precision, Recall, and Accuracy
    
    TP: C(a.clus == b.clus) AND P(a.clus == b.clus)
    FP: C(a.clus == b.clus) AND P(a.clus != b.clus)
    FN: C(a.clus != b.clus) AND P(a.clus == b.clus)
    TN: C(a.clus != b.clus) AND P(a.clus != b.clus)
    PRECISION: TP/(TP+FP)
    RECALL: TP/(TP+FN)  (sensitivity)
    ACCURACY: (TP+TN)/(TP+TN+FP+FN)
    
    :param clusters: NumPY array containing the cluster each instance is assigned
    :param true_labels: NumPY array containig the true label of each instance
    :return: Measures of accuracy, precision, and recall
    """

    same_clusters = (np.expand_dims(clusters, axis=1) == clusters)
    same_true_labels = (np.expand_dims(true_labels, axis=1) == true_labels)

    true_positives = np.triu(same_clusters & same_true_labels, 1).sum()
    false_positives = np.triu(same_clusters & ~same_true_labels, 1).sum()
    false_negatives = np.triu(~same_clusters & same_true_labels, 1).sum()
    true_negatives = np.triu(~same_clusters & ~same_true_labels, 1).sum()

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)

    return precision, recall, accuracy
    

def compute_external_indices(data, clusters, true_labels):
    (accuracy, precision, recall) = compute_precision_recall_accuracy(clusters, true_labels)

    # The homogeneity of data in the different clusters. We want to maximize. 
    # Does not work well with cluster imbalance and does not penalize having a lot of clusters
    purity = compute_purity(clusters, true_labels)

    # Adjusted version of the Rand Index (which can be seen as an accuracy in a binary classification problem). 
    # It ranges from -1 to 1, being 1 a perfect match
    adjusted_rand_index = metrics.adjusted_rand_score(true_labels, clusters)
    
    # Geometric mean between precision and recall. Ranges from 0 to 1, being 1 a good similarity between clusters
    fowlkes_mallows_score = metrics.fowlkes_mallows_score(true_labels, clusters)
    
    # Adjusted version of the Mutual Information (giving a measure of reduction of uncertainty of the obtained 
    # clusters if we know the real ones) to account for the higher values whenre there is a big number of clusters. 
    # Ranges from 0 to 1 being 1 a perfect match.
    adjusted_mutual_information_score = metrics.adjusted_mutual_info_score(true_labels, clusters)
    
    # Measures the dispersity of a true class in different clusters. 
    # Ranges from 0 to 1 being 1 when the data in a particular class falls in a single cluster
    completeness_score = metrics.completeness_score(true_labels, clusters)

    metrics_dict = { 
            'Confusion Matrix': [accuracy, precision, recall],
            'Purity': purity,
            'AdjustedRand Index':adjusted_rand_index,
            'Fowlkes Mallows':fowlkes_mallows_score,
            'Adjusted Mutual Information':adjusted_mutual_information_score,
            'Completeness Score':completeness_score,
        }
    return metrics_dict


def evaluate_resulting_clusters(data, true_labels, clusters, dataset_name, use_PCA=False):

    internal_results = compute_internal_indices(data, clusters)
    external_results = compute_external_indices(data, clusters, true_labels)

    file_path = f'{PATH_METRICS}K-Means++_metrics_{dataset_name}'
    if use_PCA:
        file_path += '_PCA'

    with open(f'{file_path}.txt', 'w') as file:

        file.write('Internal Metrics:\n\n')
        for key in internal_results:
            value = internal_results.get(key)
            file.write(f'{key}: {value}\n')

        file.write('\nExternal Metrics:\n\n')
        for key in external_results:
            value = external_results.get(key)
            if key == 'Confusion Matrix':
                file.write(f'{key}(accuracy): {value[0]}\n')
                file.write(f'{key}(precision): {value[1]}\n')
                file.write(f'{key}(recall): {value[2]}\n')
            else:
                file.write(f'{key}: {value}\n')