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

plot_vectors([x], axes=[4, 4], fname="transform_x.svg")
plot_vectors([x, y], axes=[4, 4], fname="transform_y.svg")








