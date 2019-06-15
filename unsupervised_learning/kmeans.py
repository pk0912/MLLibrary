'''
Used LLOYD'S ALGORITHM for implementing k-means clustering

Parameters to be passed while instantiating the KMeans class:
data : numpy array of dtype int64 or float64
clusters_num : number of clusters to determine (default value is 2 and value must always be of type int and greater than 1)
iterations : number of iterations for clustering result to converge (default value is 100 and must be of type int)
visualisation : type of method for visualisation : pca or tsne (default is tsne)

Use : KMeans(data, clusters_num=2, iterations=100, visualisation='pca').perform_task()

Methods :
perform_task() : returns cluster_labels of numpy array type with shape (total number of data points,) and silhouette score

TODO :
1. Better way of initialization instead of selecting only random values
2. fit, fit_transform and transform functionalities like sklearn (currently only fit_transform function is there)
'''

import os
import sys
import random
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.metrics import silhouette_score
from sklearn import decomposition
from sklearn.manifold import TSNE

MEDIA_DIRECTORY = '../media_directory'

class KMeans(object):

    def __init__(self, data, clusters_num=2, iterations=100, visualisation='tsne'):
        self.data = data
        self.cluster_labels = [0] * self.data.shape[0]
        self.clusters_num = clusters_num
        self.iterations = iterations
        self.old_centroid = []
        self.datasets = dict()
        self.new_centroid = []
        self.visualisation = visualisation.lower()

    def data_visualization(self):
        try:
            model = None
            if(self.visualisation == 'pca'):
                model = decomposition.PCA()
                model.n_components = 2
            else:
                # Setting parameters for TSNE
                tsne_para = {'perplexity': 80, 'iterations': 1000, 'learning_rate': 200}
                model = TSNE(n_components=2, random_state=0, \
                             perplexity=tsne_para['perplexity'], \
                             n_iter=tsne_para['iterations'], \
                             learning_rate=tsne_para['learning_rate'])

            # Fitting data into model for dimensionality reduction
            red_data = model.fit_transform(self.data)
            red_data = np.vstack((red_data.T, self.cluster_labels.T)).T
            red_data_df = pd.DataFrame(data=red_data, columns=('Dim_1', 'Dim_2', 'labels'))

            # Plotting clustered data
            sn.FacetGrid(red_data_df, hue='labels', height=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
            plt.savefig(os.path.join(MEDIA_DIRECTORY, 'Clustered_Datapoints_' + self.visualisation + '.svg'), format="svg")
            return True
        except Exception as e:
            print(e)
            return False

    def euclidean_distance(self, vec1, vec2):
        # Finding Euclidean distance between two vectors
        return np.sqrt(np.sum(np.square(vec1 - vec2)))

    def get_silhouette_score(self):
        # Finding Silhouette score to measure performance of the clustering algorithm on the provided dataset
        return silhouette_score(self.data, self.cluster_labels)

    def verify_user_inputs_type(self):
        # Verifying datatype of user inputs before feeding them into clustering method
        if(type(self.clusters_num) != type(0)):
            return 0
        if(type(self.iterations) != type(0)):
            return 1
        try:
            if(not(self.data.dtype == np.int64 or self.data.dtype == np.float64)):
                return 2
        except Exception as e:
            print(e)
            return 2
        return -1

    def verify_user_inputs(self):
        # Verifying values of user input before feeding them into clustering method
        type_error_status = self.verify_user_inputs_type()
        return type_error_status
        if(not(self.visualisation == 'tsne' or self.visualisation == 'pca')):
            return 3
        if(self.data.shape[0] <= 1):
            return 4
        if(self.clusters_num <= 1):
            return 5
        if(self.data.shape[0] <= self.clusters_num):
            return 6
        return 7

    def initialization(self):
        # randomly initializing old_centroid list with values from dataset
        # It would be used for assigning data points to a specific cluster for the first iteration only
        random_index = np.random.choice(self.data.shape[0], self.clusters_num, replace=False)
        self.old_centroid = [self.data[i] for i in random_index]

    def find_cluster(self):
        # Mapping each data point to the closest cluster based on distance between the data point and the centroid value
        for i in range(len(self.old_centroid)):
            self.datasets[i] = []
        for d in range(len(self.data)):
            min_dist = np.inf
            cluster = 0
            # Finding closest centroid
            for c in range(len(self.old_centroid)):
                dist = self.euclidean_distance(self.old_centroid[c], self.data[d])
                if(dist < min_dist):
                    min_dist = dist
                    cluster = c
            self.datasets[cluster].append(d)

    def calculate_centroid(self):
        # Calculating centroid for the newly formed clusters of dataset
        self.new_centroid = []
        for k in self.datasets.keys():
            centroid = np.divide(np.sum(np.array([self.data[i] for i in self.datasets[k]]), axis=(0)),len(self.datasets[k]))
            self.new_centroid.append(centroid)

    def perform_task(self):
        error_status = self.verify_user_inputs()
        if(error_status == 0):
            print('ERROR : clusters_num must be of type integer !!!')
        elif(error_status == 1):
            print('ERROR : iterations must be of type integer !!!')
        elif(error_status == 2):
            print('ERROR : data must be of type numpy.int64 or numpy.float64 !!!')
        elif(error_status == 3):
            print('ERROR : visualisation can only be tsne or pca !!!')
        elif(error_status == 4):
            print('ERROR : Number of data points for clustering must be greater than 1 !!!')
        elif(error_status == 5):
            print('ERROR : Number of clusters provided must be greater than 1 !!!')
        elif(error_status == 6):
            print('ERROR : Number of clusters must always be smaller than total number of data points!!!')
        else:
            self.initialization()
            try:
                # Iterating find_cluster and calculate_centroid method until there is no change in new centroid values
                # and old centroid values or total number of iterations are done, whichever comes first
                for i in range(self.iterations):
                    self.find_cluster()
                    self.calculate_centroid()
                    if(np.sum(np.array(self.old_centroid) - np.array(self.new_centroid)) == 0.0):
                        break
                    self.old_centroid = self.new_centroid
                for k,v in self.datasets.items():
                    for index in v:
                        self.cluster_labels[index] = k
                self.cluster_labels = np.array(self.cluster_labels)

                # Finding Silhouette score for the determined clusters
                sil_score = self.get_silhouette_score()

                # Calling the visualisation method for cluster plotting
                vis_status = self.data_visualization()
                if(not vis_status):
                    print('ERROR : Some exception occurred in plotting clusters !!!')

                print(self.cluster_labels.shape)
                return self.cluster_labels, sil_score
            except Exception as e:
                print('ERROR : ' + str(e))
                return None, None

if __name__ == '__main__':
    data = np.array([[1,2,5],[3,4,6],[9,6.0,7],[10,7,8]])
    kmeans = KMeans(data, clusters_num=2, iterations=5, visualisation='pca')
    print(kmeans.perform_task())
