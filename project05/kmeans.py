'''kmeans.py
Performs K-Means clustering
Ghailan Fadah
CS 251 Data Analysis Visualization, Spring 2022
'''
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from palettable import cartocolors


class KMeans():
    def __init__(self, data=None):
        '''KMeans constructor

        (Should not require any changes)

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''

        # k: int. Number of clusters
        self.k = None
        # centroids: ndarray. shape=(k, self.num_features)
        #   k cluster centers
        self.centroids = None
        # data_centroid_labels: ndarray. shape=(self.num_samps,)
        #   Holds index of the assigned cluster of each data sample
        self.data_centroid_labels = None

        # inertia: float.
        #   Mean squared distance between each data sample and its assigned (nearest) centroid
        self.inertia = None

        # data: ndarray. shape=(num_samps, num_features)
        self.data = data
        # num_samps: int. Number of samples in the dataset
        self.num_samps = None
        # num_features: int. Number of features (variables) in the dataset
        self.num_features = None
        if data is not None:
            self.num_samps, self.num_features = data.shape


    def set_data(self, data):
        '''Replaces data instance variable with `data`.

        Reminder: Make sure to update the number of data samples and features!

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''

        self.data = data
        self.num_features, self.num_features = data.shape
      

    def get_data(self):
        '''Get a COPY of the data

        Returns:
        -----------
        ndarray. shape=(num_samps, num_features). COPY of the data
        '''

        return np.copy(self.data)
   

    def get_centroids(self):
        '''Get the K-means centroids

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(k, self.num_features).
        '''
        return self.centroids

    def get_data_centroid_labels(self):
        '''Get the data-to-cluster assignments

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(self.num_samps,)
        '''
        return self.data_centroid_labels

    def dist_pt_to_pt(self, pt_1, pt_2):
        '''Compute the Euclidean distance between data samples `pt_1` and `pt_2`

        Parameters:
        -----------
        pt_1: ndarray. shape=(num_features,)
        pt_2: ndarray. shape=(num_features,)

        Returns:
        -----------
        float. Euclidean distance between `pt_1` and `pt_2`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        '''
        return np.sqrt(np.power((pt_1 - pt_2), 2).sum())

        
       

        
    

    def dist_pt_to_centroids(self, pt, centroids):
        '''Compute the Euclidean distance between data sample `pt` and and all the cluster centroids
        self.centroids

        Parameters:
        -----------
        pt: ndarray. shape=(num_features,)
        centroids: ndarray. shape=(C, num_features)
            C centroids, where C is an int.

        Returns:
        -----------
        ndarray. shape=(C,).
            distance between pt and each of the C centroids in `centroids`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        '''

        return np.linalg.norm(pt - centroids, axis=1)

    

    def initialize(self, k):
        '''Initializes K-means by setting the initial centroids (means) to K unique randomly
        selected data samples

        Parameters:
        -----------
        k: int. Number of clusters

        Returns:
        -----------
        ndarray. shape=(k, self.num_features). Initial centroids for the k clusters.

        NOTE: Can be implemented without any for loops
        '''

        self.k = k

        idx = np.random.choice(len(self.data), k, replace=False)
    
        centroids = self.data[idx, :] 

        self.centroids = centroids
        return centroids


    def cluster(self, k=2, tol=1e-5, max_iter=1000, verbose=False):
        '''Performs K-means clustering on the data

        Parameters:
        -----------
        k: int. Number of clusters
        tol: float. Terminate K-means if the (absolute value of) the difference between all 
        the centroid values from the previous and current time step < `tol`.
        max_iter: int. Make sure that K-means does not run more than `max_iter` iterations.
        verbose: boolean. Print out debug information if set to True.

        Returns:
        -----------
        self.inertia. float. Mean squared distance between each data sample and its cluster mean
        int. Number of iterations that K-means was run for

        TODO:
        - Initialize K-means variables
        - Do K-means as long as the max number of iterations is not met AND the absolute value of the
        difference between the previous and current centroid values is > `tol`
        - Set instance variables based on computed values.
        (All instance variables defined in constructor should be populated with meaningful values)
        - Print out total number of iterations K-means ran for
        '''


        cen = self.initialize(k)

        max = max_iter
        diff = np.array([1])
        turn = 0
        while max > 0 and abs(diff.sum()) > tol:
            turn += 1
            ids = self.assign_labels(cen)
            cen, diff = self.update_centroids(k, ids, cen)
            ineria = self.compute_inertia()
            max -= 1

        return ineria, turn



       

    def cluster_batch(self, k=2, n_iter=1, verbose=False):
        '''Run K-means multiple times, each time with different initial conditions.
        Keeps track of K-means instance that generates lowest inertia. Sets the following instance
        variables based on the best K-mean run:
        - self.centroids
        - self.data_centroid_labels
        - self.inertia

        Parameters:
        -----------
        k: int. Number of clusters
        n_iter: int. Number of times to run K-means with the designated `k` value.
        verbose: boolean. Print out debug information if set to True.
        '''

        list_inertia = []
        list_cen = []
        list_labels = []

        for i in range(n_iter):
            inertia, turns = self.cluster(k)
            list_inertia.append(inertia)
            list_cen.append(self.centroids)
            list_labels.append(self.data_centroid_labels)

        minu = min(list_inertia)
        index = list_inertia.index(minu)

        self.inertia = minu
        self.centroids = list_cen[index]
        self.data_centroid_labels = list_labels[index]
            

    def assign_labels(self, centroids):
        '''Assigns each data sample to the nearest centroid

        Parameters:
        -----------
        centroids: ndarray. shape=(k, self.num_features). Current centroids for the k clusters.

        Returns:
        -----------
        ndarray. shape=(self.num_samps,). Holds index of the assigned cluster of each data sample

        Example: If we have 3 clusters and we compute distances to data sample i: [0.1, 0.5, 0.05]
        labels[i] is 2. The entire labels array may look something like this: [0, 2, 1, 1, 0, ...]
        '''

        labels = np.empty([self.num_samps])
        for i in range(self.num_samps):
            dis = self.dist_pt_to_centroids(self.data[i, :], centroids)
            c = np.argmin(dis)
            labels[i] = c


        self.data_centroid_labels = np.array(labels, np.int64)

        return self.data_centroid_labels
    
    def update_centroids(self, k, data_centroid_labels, prev_centroids):
        '''Computes each of the K centroids (means) based on the data assigned to each cluster
        
        The basic algorithm is to loop through each cluster and assign the mean value of all 
        the points in the cluster. If you find a cluster that has 0 points in it, then you should
        choose a random point from the data set and use that as the new centroid.


        Parameters:
        -----------
        k: int. Number of clusters
        data_centroid_labels. ndarray. shape=(self.num_samps,)
            Holds index of the assigned cluster of each data sample
        prev_centroids. ndarray. shape=(k, self.num_features)
            Holds centroids for each cluster computed on the PREVIOUS time step

        Returns:
        -----------
        new_centroids. ndarray. shape=(k, self.num_features).
            Centroids for each cluster computed on the CURRENT time step
        centroid_diff. ndarray. shape=(k, self.num_features).
            Difference between current and previous centroid values
        '''
       
        new_centers = np.empty(prev_centroids.shape)
        for i in range(k):
            if self.data[data_centroid_labels == i].size == 0:
                idx = np.random.choice(len(self.data), 1, replace=False)
                centroid = self.data[idx, :] 
                new_centers[i] = centroid
            else:

                new_centers[i] = np.mean(self.data[data_centroid_labels == i], axis = 0)

        diff = new_centers - prev_centroids
        self.centroids = new_centers
        return (new_centers, diff)

      


        pass

    def compute_inertia(self):
        '''Mean squared distance between every data sample and its assigned (nearest) centroid

        Parameters:
        -----------
        None

        Returns:
        -----------
        float. The average squared distance between every data sample and its assigned cluster centroid.
        '''
        c = 0
        for i in range(self.num_samps):
            dis = self.dist_pt_to_pt(self.data[i, :], self.centroids[self.data_centroid_labels[i], :])**2
            c+= dis

        self.inertia = (1/self.num_samps)* c
        return (1/self.num_samps)* c

    def plot_clusters(self):
        '''Creates a scatter plot of the data color-coded by cluster assignment.

        TODO:
        - Plot samples belonging to a cluster with the same color.
        - Plot the centroids in black with a different plot marker.
        - The default scatter plot color palette produces colors that may be difficult to discern
        (especially for those who are colorblind). Make sure you change your colors to be clearly
        differentiable.
            You should use a palette Colorbrewer2 palette. Pick one with a generous
            number of colors so that you don't run out if k is large (e.g. 10).
        '''
        color_map = cartocolors.qualitative.Bold_10.mpl_colors
        for i in range(self.k):
            w0 = self.data[self.data_centroid_labels == i]
            plt.scatter(w0[:,0], w0[:,1], color = color_map[i])

        plt.plot(self.centroids[:, 0], self.centroids[:, 1], 'k*',)

        plt.show()



        pass

    def elbow_plot(self, max_k, n_iter = 1):
        '''Makes an elbow plot: cluster number (k) on x axis, inertia on y axis.

        Parameters:
        -----------
        max_k: int. Run k-means with k=1,2,...,max_k.
        n_iter: number of times to run cluster

        TODO:
        - Run k-means with k=1,2,...,max_k, record the inertia.
        - Make the plot with appropriate x label, and y label, x tick marks.
        '''

        y_array = np.empty(max_k)

        for i in range(max_k):
            self.cluster_batch(i+1, n_iter)
            y_array[i] = self.inertia

        plt.plot(np.arange(max_k) + 1, y_array)
        plt.xlabel("number of clusters")
        plt.ylabel("inertia")
        plt.xticks(np.arange(1, max_k + 1, 1))
        plt.show()


        pass

    def replace_color_with_centroid(self):
        '''Replace each RGB pixel in self.data (flattened image) with the closest centroid value.
        Used with image compression after K-means is run on the image vector.

        Parameters:
        -----------
        None

        Returns:
        -----------
        None
        '''

        for i in range(0,self.k):
            ckuster = np.where(self.data_centroid_labels == i)[0]
            self.data[ckuster] = self.centroids[i]

    def siloutte_plot(self, max_k):
        costs = []
        for p in range(max_k):
            k_m, turn= self.cluster(p +1,)

            dist_ji = 0
            a = 1
            s=0
            for i in range(len(self.data[0])):
                    for j in range(p):
                        dist_ji += self.dist_pt_to_pt(self.centroids[j,:],self.data[i,:])

            s = (dist_ji - a)/max(dist_ji,a)
            s = np.array(s)
            s =  s.item()
            costs.append(s)
        x = np.arange(max_k)
        plt.plot(x,costs)
        plt.title("Silhoutte Score")
        plt.xlabel("Number of Ks")
        plt.ylabel("Dispersion")


        
