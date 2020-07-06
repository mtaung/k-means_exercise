import seaborn as sns 
import matplotlib.pyplot as plt 
import numpy as np

def plot_2d_clusters(data, centroids, file_out):
    """
    Plots and saves a scattergraph of some data and cluster vectors.
    Input data will be transposed to plot along the columns (features/dimensions).
    Input centroids will be converted into an ndarray and transposed similarly for plotting along columns. 
    Files are saved by default to the subdirectory /figures/

    Args:
        data (ndarray): some input data where each row is an observation/event.
        centroids (list): a list of arrays where each array holds a single centroid vector. 

    Returns:
        None
    """
    plt.clf()
    data_transpose = data.T
    centroids_transpose = np.vstack(centroids).T
    sns.scatterplot(data_transpose[0], data_transpose[1])
    sns.scatterplot(centroids_transpose[0], centroids_transpose[1])
    plt.savefig(f'figures/{file_out}')