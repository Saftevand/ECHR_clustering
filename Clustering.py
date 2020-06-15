import DataLoader
import numpy as np
import pickle
from sklearn.cluster import AffinityPropagation, KMeans
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import sklearn
from sklearn import preprocessing
from time import time
from sklearn.pipeline import make_pipeline


def run_affinity_propagation(sparse_matrix):
    AffinityPropagation.fit(sparse_matrix)


def calc_adjacency_matrix(data, similarity_measure='euclidean'):
    adjacency_matrix = []


def calc_euclidean_distance(matrix):
    similarities = euclidean_distances(matrix)
    return similarities

def calc_cosine_similarity(matrix):
    similarities = cosine_similarity(matrix)
    return similarities


def cluster_LCA_kmeans(matrix):

    print("Performing dimensionality reduction using LSA")
    t0 = time()
    # Vectorizer results are normalized, which makes KMeans behave as
    # spherical k-means for better results. Since LSA/SVD results are
    # not normalized, we have to redo the normalization.
    svd = sklearn.decomposition.TruncatedSVD(100)
    normalizer = preprocessing.Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    X = lsa.fit_transform(matrix)

    print("done in %fs" % (time() - t0))

    explained_variance = svd.explained_variance_ratio_.sum()
    print("Explained variance of the SVD step: {}%".format(
        int(explained_variance * 100)))

    print()

    # #############################################################################
    # Do the actual clustering


    km = KMeans(n_clusters=10, init='k-means++', max_iter=100, n_init=1, verbose=True)
    km.fit(X)
    return km.labels_, km.cluster_centers_, X