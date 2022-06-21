# Parts of speech tagging - Working with tags and Numpy
import math
import numpy as np
import pandas as pd


def print_matrix(_matrix, _tags):
    '''

    :param _matrix:
    :param _tags:
    :return:
    '''
    print(pd.DataFrame(_matrix, index=_tags, columns=_tags))


# Define tags for Adverb, Noun and To (the preposition) , respectively
tags = ['RB', 'NN', 'TO']

# Define 'transition_counts' dictionary
# Note: values are the same as the ones in the assignment
transition_counts = {
    ('NN', 'NN'): 16241,
    ('RB', 'RB'): 2263,
    ('TO', 'TO'): 2,
    ('NN', 'TO'): 5256,
    ('RB', 'TO'): 855,
    ('TO', 'NN'): 734,
    ('NN', 'RB'): 2431,
    ('RB', 'NN'): 358,
    ('TO', 'RB'): 200
}

num_tags = len(tags)
transition_matrix = np.zeros((num_tags, num_tags))

sorted_tags = sorted(tags)
for i in range(num_tags):
    for j in range(num_tags):
        tag_tuple = (sorted_tags[i], sorted_tags[j])
        transition_matrix[i, j] = transition_counts.get(tag_tuple)

row_sum = transition_matrix.sum(axis=1, keepdims=True)
transition_matrix = transition_matrix/10
transition_matrix = transition_matrix / row_sum
print_matrix(transition_matrix, sorted_tags)

# Copy transition matrix for numpy functions example
t_matrix_for = np.copy(transition_matrix)
t_matrix_np = np.copy(transition_matrix)

for i in range(num_tags):
    t_matrix_for[i, i] = t_matrix_for[i, i] + math.log(row_sum[i])

print(t_matrix_for)

mat_diagonal = np.diagonal(t_matrix_np).reshape(row_sum.shape)
mat_diagonal = mat_diagonal + np.vectorize(math.log)(row_sum)
np.fill_diagonal(t_matrix_np, mat_diagonal)

print(t_matrix_np)