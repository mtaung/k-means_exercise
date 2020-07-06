from clustering.centroid import init_centroid, assign_clusters, adjust_centroids
from clustering.viz import plot_2d_clusters
import clustering.toolbox

def kmeans_cluster(data, k, n_iter):
    """
    Carries out a simple k-means clustering sequence for a specified number of iterations. 

    Args:
        data (ndarray): some input data where each row is an observation/event.
        k (int): the k number of specified centroids to initialise.
        n_iter (int): the number of iterations to compute and adjust centroids.

    Returns: 
        assignments (list): A list of arrays where each array represents a cluster, and contains its constituent members. 
    """
    centroids = init_centroid(data, k)
    for i in range(n_iter):
        plot_2d_clusters(data, centroids, f'iter_{i}')
        assignments = assign_clusters(data, centroids) 
        centroids = adjust_centroids(centroids, assignments)   
    return assignments