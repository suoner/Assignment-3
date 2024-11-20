import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris 
# from sklearn.metrics import  
from cluster import (KMeans)
from cluster.visualization import plot_3d_clusters


def main(): 
    # Set random seed for reproducibility
    np.random.seed(42)
    # Using sklearn iris dataset to train model
    og_iris = np.array(load_iris().data)
    
    # Initialize your KMeans algorithm
    kmeans = KMeans(k=3, metric="euclidean", max_iter=300, tol=1e-4)
    
    # Fit model
    kmeans.fit(og_iris)
 

    # Load new dataset
    df = np.array(pd.read_csv('data/iris_extended.csv', 
                usecols = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']))

    # Predict based on this new dataset
    predictions = kmeans.predict(df)   
    print("Centroids:\n", kmeans.get_centroids())
    print("Inertia (Error):", kmeans.get_error())
    
    # You can choose which scoring method you'd like to use here:
    score = kmeans.get_error()
    print("Inertia (Error):", score) 
    
    # Plot your data using plot_3d_clusters in visualization.py
    plot_3d_clusters(df, predictions, kmeans.get_centroids(), score)

    
    # Try different numbers of clusters
    inertia_values = []
    cluster_range = range(1, 11)
    
    for k in cluster_range:
        temp_kmeans = KMeans(k=k, metric="euclidean", max_iter=300, tol=1e-4)
        temp_kmeans.fit(og_iris)
        inertia_values.append(temp_kmeans.get_error())
    
    # Plot the elbow plot
    plt.figure(figsize=(8, 6))
    plt.plot(cluster_range, inertia_values, marker='o', label="Inertia")
    plt.title('Elbow Plot')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.grid()
    plt.legend()
    plt.show()
    
    # Question: 
    # Please answer in the following docstring how many species of flowers (K) you think there are.
    # Provide a reasoning based on what you have done above: 
    
    """
    How many species of flowers are there:  3
    
    Reasoning: The elbow at k=3 in the elbow plot indicates that 3 clusters are close to optimal, and adding more clusters marginally improves compactness.
    Also, the scatter plot shows that the data clusters seem aligned with the natural groupings.
    
    """

    
if __name__ == "__main__":
    main()