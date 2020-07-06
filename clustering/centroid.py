import numpy as np
from numpy.linalg import norm
from clustering import toolbox

def init_centroid(input_data, k):
    """
    Initialises centroids within the bounds of the input data.

    Args:
        input_data (ndarray): some input data where each row is an observation/event.
        k (int): the k number of specified centroids to initialise.

    Returns:
        centroids (ndarray): an array where each row is a centroid vector.
    """
    bounds = toolbox.get_bounds(input_data)
    centroids = []
    for i in range(k):
        centroids.append([np.random.randint(i[0], i[1]) for i in bounds])
    return np.array(centroids)
    
def assign_clusters(x, centroids):
    """
    Assigns each observation to the closest cluster. 
    The closest cluster is determined by minimising the euclidean distance. 

    Args: 
        x (ndarray): a row vector that contains a single observation.
        centroids (ndarray): an array where each row is a cluster vector.

    Returns:
        labels (list): A list of arrays where each array represents a cluster and its rows comprise the constituent observations.
    """
    labels = [[] for i in centroids]
    for x_i in x:
        distances = []
        for centroid in centroids:
            distances.append([norm(x_i-centroid)**2])
        min_index = np.argmin(distances)
        labels[min_index].append(x_i)
    labels = [np.array(i) for i in labels]
    return labels

def adjust_centroids(centroids, assignments):
    """
    Adjusts some centroid vectors towards the means of their constituent observations.

    Args:
        centroids (ndarray): an array where each row is a centroid vector.
        assignments (list): a list of arrays where each array contains the constituent members of a centroid/cluster.
    
    Returns:
        adjusted_centroids (list): a list of arrays where each array contains a centroid vector. 
    """
    old_assignments = assignments 
    adjusted_centroids = []
    for centroid_set in old_assignments:
        new_centroid = np.sum(centroid_set, axis=0)/len(centroid_set)
        adjusted_centroids.append(new_centroid)
    return adjusted_centroids