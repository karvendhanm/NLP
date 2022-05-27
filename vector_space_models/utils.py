import numpy as np
from operator import itemgetter
import pickle

fh = open('./data/Files/word_embeddings_subset.p', 'rb')
word_embeddings = pickle.load(fh)

def euclidean_distance(vector1, vector2):
    '''

    :param vector1:
    :param vector2:
    :return:
    '''
    return np.linalg.norm(vector1 - vector2)


def cosine_similarity(vector1, vector2):
    '''

    :param vector1:
    :param vector2:
    :return:
    '''
    numerator = np.dot(vector1, vector2)
    denominator = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    return numerator / denominator

def get_word_vector(embeddings, word):
    '''

    :param embeddings:
    :param word:
    :return:
    '''
    return embeddings[word]

def get_closest_vector(embeddings, vector, *args):
    '''

    :param embeddings:
    :param vector:
    :param args:
    :return:
    '''

    similarity = -1
    country = ''
    for key in embeddings.keys():
        if key not in args:
            score = cosine_similarity(embeddings[key], vector)

            if score > similarity:
                similarity = score
                country = key

    return country


def get_country(city1, country1, city2, embeddings, cosine_similarity=cosine_similarity):
    """
    Input:
        city1: a string (the capital city of country1)
        country1: a string (the country of capital1)
        city2: a string (the capital city of country2)
        # CODE REVIEW COMMENT: Embedding incomplete code comment, should add "and values are their emmbeddings"
        embeddings: a dictionary where the keys are words and
    Output:
        countries: a dictionary with the most likely country and its similarity score
    """
    city1_vec = get_word_vector(embeddings, city1)
    city2_vec = get_word_vector(embeddings, city2)
    country1_vec = get_word_vector(embeddings, country1)

    intermediate_vector = country1_vec - city1_vec
    country2_vec_pred = city2_vec + intermediate_vector

    country = get_closest_vector(embeddings, country2_vec_pred, city1, country1, city2)
    return country

def compute_pca(X, n_components=2):
    """
    Input:
        X: of dimension (m,n) where each row corresponds to a word vector
        n_components: Number of components you want to keep.
    Output:
        X_reduced: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """

    ### START CODE HERE ###
    # mean center the data
    X_demeaned = X - np.mean(X, axis=0).reshape(1, -1)

    # calculate the covariance matrix
    covariance_matrix = np.cov(X_demeaned.T)

    # calculate eigenvectors & eigenvalues of the covariance matrix
    eigen_vals, eigen_vecs = np.linalg.eigh(covariance_matrix)

    # sort eigenvalue in increasing order (get the indices from the sort)
    idx_sorted = np.argsort(eigen_vals)

    # reverse the order so that it's from highest to lowest.
    idx_sorted_decreasing = idx_sorted[::-1]

    # sort the eigen values by idx_sorted_decreasing
    eigen_vals_sorted = eigen_vals[idx_sorted_decreasing]

    # sort eigenvectors using the idx_sorted_decreasing indices
    eigen_vecs_sorted = eigen_vecs[:, idx_sorted_decreasing]

    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    eigen_vecs_subset = eigen_vecs_sorted[:, :n_components]

    # transform the data by multiplying the transpose of the eigenvectors with the transpose of the de-meaned data
    # Then take the transpose of that product.
    X_reduced = np.dot(X_demeaned, eigen_vecs_subset)

    ### END CODE HERE ###

    return X_reduced













