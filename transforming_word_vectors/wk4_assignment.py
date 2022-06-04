# Naive Machine translation and LSH

import pickle
import string

import time

import nltk
import numpy as np
from nltk.corpus import stopwords, twitter_samples

from transforming_word_vectors.utils_nb import (get_dict, process_tweet, cosine_similarity, get_matrices, compute_loss)
from transforming_word_vectors.utils_nb import compute_gradient, align_embeddings, nearest_neighbor, test_vocabulary
from os import getcwd
en_embeddings_subset = pickle.load(open('./data/Files/en_embeddings.p', 'rb'))
fr_embeddings_subset = pickle.load(open('./data/Files/fr_embeddings.p', 'rb'))

en_fr_train = get_dict('./data/Files/en-fr.train.txt')
print('The length of the English to French training dictionary is', len(en_fr_train))
en_fr_test = get_dict('./data/Files/en-fr.test.txt')
print('The length of the English to French testing dictionary is', len(en_fr_test))

X_train, Y_train = get_matrices(en_fr_train, fr_embeddings_subset, en_embeddings_subset)

# Translations:
# Translation as linear transformation of embeddings
# will create a transformation matrix to convert english vector to french vector using gradient descent
# and using square of frobenius as loss function.

np.random.seed(123)
m = 10
n = 5
X = np.random.rand(m, n)
Y = np.random.rand(m, n) * 0.1
R = np.random.rand(n, n)

loss = compute_loss(X, Y, R)
print(f'Exected loss for an experiment with random matrices: {loss:.4f}')

# finding the gradient
# Testing your implementation.
np.random.seed(123)
m = 10
n = 5
X = np.random.rand(m, n)
Y = np.random.rand(m, n) * .1
R = np.random.rand(n, n)
gradient = compute_gradient(X, Y, R)
print(f"First row of the gradient matrix: {gradient[0]}")

# Finding the optimal R with gradient descent algorithm.
R_train = align_embeddings(X_train, Y_train, train_steps=400, learning_rate=0.8)

v = np.array([1, 0, 1])
candidates = np.array([[1, 0, 5], [-2, 5, 3], [2, 0, 1], [6, -9, 5], [9, 9, 9]])
nearest_neighbor(v, candidates, 3)

X_test, Y_test = get_matrices(en_fr_test, fr_embeddings_subset, en_embeddings_subset)
# test your translation and compute its accuracy.
accuracy = test_vocabulary(X_test, Y_test, R_train)
print(accuracy)

