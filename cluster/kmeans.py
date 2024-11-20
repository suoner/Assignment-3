import numpy as np
from scipy.spatial.distance import cdist

class KMeans():
    def __init__(self, k: int, metric:str, max_iter: int, tol: float):
        """
        Args:
            k (int): 
                The number of centroids you have specified. 
                (number of centroids you think there are)
                
            metric (str): 
                The way you are evaluating the distance between the centroid and data points.
                (euclidean)
                
            max_iter (int): 
                The number of times you want your KMeans algorithm to loop and tune itself.
                
            tol (float): 
                The value you specify to stop iterating because the update to the centroid 
                position is too small (below your "tolerance").
        """ 
        self.k = k
        self.metric = metric
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.error = np.inf

    
    def fit(self, matrix: np.ndarray):
        """
        This method takes in a matrix of features and attempts to fit your created KMeans algorithm
        onto the provided data 
        
        Args:
            matrix (np.ndarray): 
                This will be a 2D matrix where rows are your observation and columns are features.
                
                Observation meaning, the specific flower observed.
                Features meaning, the features the flower has: Petal width, length and sepal length width 
        """
        # Randomly initialize centroids
        random_indices = np.random.choice(matrix.shape[0], self.k, replace=False)
        self.centroids = matrix[random_indices]

        for _ in range(self.max_iter):
            # Calculate Distances between each point and centroids
            distances = cdist(matrix, self.centroids, metric=self.metric)

            # Assign points to the nearest centroid
            labels = np.argmin(distances, axis=1)

            # Update centroids based on the averge of assigned points
            new_centroids = np.array([matrix[labels == i].mean(axis=0) if len(matrix[labels == i]) > 0 else self.centroids[i]
                                       for i in range(self.k)])
            
            # Calculate inertia (sum of squared distances to closest centroid)
            inertia = np.sum((cdist(matrix, new_centroids, metric=self.metric).min(axis=1)) ** 2)

            # Check for convergence
            if np.abs(self.error - inertia) < self.tol:
                break

            self.centroids = new_centroids
            self.error = inertia

    
    def predict(self, matrix: np.ndarray) -> np.ndarray:
        """
        Predicts which cluster each observation belongs to.
        Args:
            matrix (np.ndarray): 
                

        Returns:
            np.ndarray: 
                An array/list of predictions will be returned.
        """
        distances = cdist(matrix, self.centroids, metric=self.metric)
        return np.argmin(distances, axis=1) 
    
    def get_error(self) -> float:
        """
        The inertia of your KMeans model will be returned

        Returns:
            float: 
                inertia of your fit

        """
        return self.error
    
    def get_centroids(self) -> np.ndarray:
    
        """
        Your centroid positions will be returned. 
        """
        return self.centroids
        
    