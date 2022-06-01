# Basic hash tables

import numpy as np
from transforming_word_vectors.utils_nb import plot_vectors
import matplotlib.pyplot as plt


def basic_hash_table(value_l, n_buckets):
    def hash_function(value, n_buckets):
        return int(value) % n_buckets

    hash_table = {val: [] for val in range(n_buckets)}

    for value in value_l:
        hash_value = hash_function(value, n_buckets)
        hash_table[hash_value].append(value)

    return hash_table


value_l = [100, 10, 14, 17, 97]
hash_table_example = basic_hash_table(value_l, n_buckets=10)
print(hash_table_example)

# planes
P = np.array([[1, 2]])  # defining a single plane by its normal vector.
rot_deg = 90
angle = rot_deg * (np.pi / 180)
rotation_matrix = [[np.cos(angle), np.sin(angle)],
                   [-np.sin(angle), np.cos(angle)]]

P_rot = np.dot(P, rotation_matrix)  # rotate the normal vector anti-clock wise to align with the plane

fig, ax1 = plt.subplots(figsize=(8, 8))
plot_vectors([P], colors=['b'], axes=[2, 2], ax=ax1)

plot_vectors([P_rot * 4, P_rot * -4], colors=['k', 'k'], axes=[4, 4], ax=ax1)

for i in range(10):
    v1 = np.random.uniform(-2, 2, 2)
    side_of_plane = np.sign(np.dot(P, v1.T))

    if side_of_plane == 1:
        ax1.plot([v1[0]], [v1[1]], 'bo')
    else:
        ax1.plot([v1[0]], [v1[1]], 'ro')

plt.show()

P = np.array([[1, 1]])  # Single plane
v1 = np.array([[1, 2]])  # Sample point 1
v2 = np.array([[-1, 1]])  # Sample point 2
v3 = np.array([[-2, -1]])  # Sample point 3

np.dot(P, v1.T)
np.dot(P, v2.T)
np.dot(P, v3.T)


def side_of_plane(P, v):
    dotproduct = np.dot(P, v.T)  # Get the dot product P * v'
    sign_of_dot_product = np.sign(dotproduct)  # The sign of the elements of the dotproduct matrix
    sign_of_dot_product_scalar = sign_of_dot_product.item()  # The value of the first item
    return sign_of_dot_product_scalar


side_of_plane(P, v1)  # In which side is [1, 2]
side_of_plane(P, v2)  # In which side is [-1, 1]
side_of_plane(P, v3)  # In which side is [-2, -1]

# Hash functions with multiple planes
P1 = np.array([[1, 1]])  # First plane 2D
P2 = np.array([[-1, 1]])  # Second plane 2D
P3 = np.array([[-1, -1]])  # Third plane 2D
P_l = [P1, P2, P3]  # List of arrays. It is the multi plane

# Vector to search
v = np.array([[2, 2]])


def hash_multi_planes(P_l, v):
    hash_value = 0
    for _idx, plane in enumerate(P_l):
        sign = side_of_plane(plane, v)
        hash_i = 1 if sign >= 0 else 0
        hash_value += (2 ** _idx * hash_i)
    return hash_value


hash_multi_planes(P_l, v)

# Random planes
np.random.seed(0)
num_dimensions = 2
num_planes = 3
random_planes_matrix = np.random.normal(size=(num_planes, num_dimensions))

v = np.array([[2, 2]])


def side_of_plane_matrix(P, v):
    dotProduct = np.dot(P, v.T)
    sign_of_dot_product = np.sign(dotProduct)
    return sign_of_dot_product


sides_l = side_of_plane_matrix(random_planes_matrix, v)
print(sides_l)

def hash_multi_plane_matrix(P, v, num_of_planes):
    sides_matrix = side_of_plane_matrix(P, v)
    hash_value = 0
    for plane in range(num_of_planes):
        sign = sides_matrix[plane].item()
        hash_i = 1 if sign >= 0 else 0
        hash_value += (2**plane) * hash_i
    return hash_value

hash_multi_plane_matrix(random_planes_matrix, v, num_planes)
