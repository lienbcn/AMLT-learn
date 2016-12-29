"""
.. module:: SeededKMeans

SeededKmeans
*************

:Description: SeededKmeans is the implementation of the Seeded-KMeans algorithm described in the paper
Semi-supervised Clustering by Seeding

It uses labelled data to perform an initialization of the clusters in the k-means clustering


"""

import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import check_random_state
from sklearn.cluster import k_means
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.datasets import make_blobs
import itertools
import os
import pandas as pd


class SeededKMeans(BaseEstimator, ClusterMixin):
    """
    Seeded K-means
    """

    def __init__(self, n_clusters=3, max_iter=50, tol=1e-3, verbose=0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.centroids_ = None # Save here the centroids size (n_clusters, n_dimensions)
        self.labels_ = None # Save here the labels (cluster index) of each instance, size (n_instances)

    def get_random_centroid(self, X):
        maxRange = np.max(X, 0)
        minRange = np.min(X, 0)
        nDimensions = X.shape[1]
        randomCentroid = np.random.rand(nDimensions) # Initialize
        for i in range(nDimensions):
            maxVal = maxRange[i]
            minVal = minRange[i]
            randomCentroid[i] = (maxVal - minVal) * np.random.rand() + minVal
        return randomCentroid

    def _initialize_by_seeds(self, X, Y):
        n_dimensions = X.shape[1]
        self.centroids_ = np.zeros((self.n_clusters, n_dimensions))
        aClasses = np.unique(Y[Y >= 0]) # -1 means no class
        n_classes = aClasses.shape[0]
        
        # Check that the classes go from 0 to num_classes-1:
        if not np.array_equal(range(n_classes), aClasses):
            raise ValueError('Incorrect classes provided. They should be sequential integers')

        for iClass in aClasses:
            instancesThisClass = X[Y == iClass, :]
            centroidThisClass = np.mean(instancesThisClass, 0) # shape = (1, n_dimensions)
            self.centroids_[iClass, :] = centroidThisClass

        # Initialize random clusters to the centroids that do not have seed:
        nSeeds = len(aClasses)
        for iCentroidIndex in range(nSeeds, self.n_clusters):
            self.centroids_[iCentroidIndex] = self.get_random_centroid(X)

    def fit(self, X, Y):
        n_instances = X.shape[0]
        self._initialize_by_seeds(X, Y)
        self.X_fit_ = X
        self.distances_ = np.zeros((n_instances, self.n_clusters))
        self.labels_ = np.zeros(n_instances)

        for it in xrange(self.max_iter):
            self._compute_distances()
            labels_old = self.labels_
            self.labels_ = self.distances_.argmin(axis=1)

            # Compute the number of samples whose cluster did not change
            # since last iteration.
            n_same = np.sum((self.labels_ - labels_old) == 0)
            if 1 - float(n_same) / n_instances < self.tol:
                if self.verbose:
                    print("Converged at iteration", it + 1)
                break

        return self

    def _compute_distances(self):
        """Compute a n_instances x n_clusters distance matrix using euclidean distance"""
        for c in xrange(self.n_clusters):
            centroid = self.centroids_[c, :]
            for i in range(self.X_fit_.shape[0]):
                instance = self.X_fit_[i, :]
                self.distances_[i, c] = np.linalg.norm(centroid - instance)
    

    def predict(self, X):
        n_instances = X.shape[0]
        self._compute_distances()
        return self.distances_.argmin(axis=1)


def get_prediction_accuracy(y, labels):
    n_instances = y.shape[0]
    correctLabels = np.sum(labels == y)
    return float(correctLabels)/float(n_instances)

# TODO: http://things-about-r.tumblr.com/post/36087795708/matching-clustering-solutions-using-the-hungarian
def find_correct_predictions(y, labels):
    n_clusters = len(np.unique(labels))
    bestPrediction = labels
    for mapping in itertools.permutations(range(n_clusters)):
        predictedLabels = [mapping[label] for label in labels]
        if get_prediction_accuracy(y, predictedLabels) > get_prediction_accuracy(y, bestPrediction):
            bestPrediction = predictedLabels
    print('Correct predictions found')
    return bestPrediction

def testGeneratedData():
    X, y = make_blobs(n_samples=1000, centers=np.array([[-2.5, 0], [2.5, 0], [0, 2.5], [0, -2.5]]))

    n_clusters = np.unique(y).shape[0]
    n_instances = X.shape[0]

    sm = SeededKMeans(n_clusters=n_clusters, max_iter=2000, verbose=1)
    sm.fit(X, y)
    predictedLabels = sm.predict(X)
    trueLabels = y

    adjustedRandScore = metrics.adjusted_rand_score(trueLabels, predictedLabels)
    print('Seeded KMeans adjusted rand score: %s' % (adjustedRandScore))

    (centers, predictedLabels, inertia, best_n_iter) = k_means(X, n_clusters, n_init=1, return_n_iter=True)
    print('Number of iterations: %s' % (best_n_iter))
    # Find the correct predictions, the permutations that maximizes the accuracy:
    # predictedLabels = kmeans.labels_
    trueLabels = y
    
    adjustedRandScore = metrics.adjusted_rand_score(trueLabels, predictedLabels)
    print('KMeans adjusted rand score: %s' % (adjustedRandScore))



    predictedLabels = find_correct_predictions(trueLabels, predictedLabels);
    tColour = tuple([0, 1, 0])
    plt.scatter(X[y == predictedLabels, 0], X[y == predictedLabels, 1], c=tColour, alpha=0.5)
    tColour = tuple([1, 0, 0])
    plt.scatter(X[y != predictedLabels, 0], X[y != predictedLabels, 1], c=tColour, alpha=0.5)
    plt.show()

def testRealData():
    sDirname = os.path.dirname(os.path.abspath(__file__))
    dfAspen = pd.read_csv(os.path.join(sDirname, '..', 'datasets', 'aspen.csv'), ';')
    dfAspen = dfAspen.dropna()
    dfAspen = dfAspen.reindex(np.random.permutation(dfAspen.index))

    asCities = dfAspen['address_city'].unique()
    X = dfAspen[['location_coordinates_0', 'location_coordinates_1']].as_matrix()
    Y = dfAspen['address_city'].as_matrix()
    Y = np.array([asCities.tolist().index(sCity) for sCity in Y]) # Convert array of strings to sequential integers
    # Convert coordinates to x,y grid using the Equirectangular projection
    X = X * np.pi / 180
    fMeanLatitude = X.mean(0)[0]
    X[:, 0] = X[:, 0] * np.cos(fMeanLatitude)
    # First x, then y:
    Xaux = np.copy(X)
    X[:, 1] = Xaux[:, 0]
    X[:, 0] = Xaux[:, 1]

    n_clusters = len(asCities)

    # K-means:
    (centers, PredictedLabels, inertia, best_n_iter) = k_means(X, n_clusters, n_init=1, return_n_iter=True)
    adjustedRandScore = metrics.adjusted_rand_score(Y, PredictedLabels)
    print('KMeans adjusted rand score: %s' % (adjustedRandScore))
    print('Number of iterations: %s' % (best_n_iter))

    # Seeded k-means with all seeds:
    # Drop some seeds:
    fRatio = 0
    dMapping = {}
    maxSeed = -1
    SomeSeeds = np.repeat(-1, len(Y))
    for i in range(len(Y)):
        if np.random.rand() > fRatio:
            continue
        iCity = Y[i]
        if iCity not in dMapping:
            maxSeed += 1
            dMapping[iCity] = maxSeed
        SomeSeeds[i] = dMapping[iCity]

    sm = SeededKMeans(n_clusters=n_clusters, max_iter=2000, verbose=1)
    sm.fit(X, SomeSeeds)
    PredictedLabels = sm.predict(X)
    adjustedRandScore = metrics.adjusted_rand_score(Y, PredictedLabels)
    print('Seeded KMeans adjusted rand score: %s' % (adjustedRandScore))

    for predictedLabel in np.unique(PredictedLabels):
        plt.scatter(X[PredictedLabels == predictedLabel, 0], X[PredictedLabels == predictedLabel, 1], color=(np.random.rand(), np.random.rand(), np.random.rand()), alpha=1)
    plt.show()


if __name__ == '__main__':
    testRealData()


