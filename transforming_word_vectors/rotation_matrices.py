import numpy as np
import matplotlib.pyplot as plt

from transforming_word_vectors.utils_nb import plot_vectors

# create a 2X2 matrix
R = np.array([
    [2, 0],
    [0, -2]
])

x = np.array([[1, 1]])
y = np.dot(x, R)

plot_vectors([x, y], axes=[4, 4], fname='transformx_and_y.svg')

angle = 100 * (np.pi / 180)

Ro = np.array([[np.cos(angle), -np.sin(angle)],
               [np.sin(angle), np.cos(angle)]])

x2 = np.array([2, 2]).reshape(1, -1)  # make it a row vector
y2 = np.dot(Ro, x2.T).reshape(1, -1)

plot_vectors([x2, y2])