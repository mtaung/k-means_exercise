import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from clustering.cluster import kmeans_cluster

data = np.genfromtxt('data/g2-2-30.csv', dtype='int', delimiter=',')
clusters = kmeans_cluster(data, 2, 5)
