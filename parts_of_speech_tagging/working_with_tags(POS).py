# Parts of speech tagging - Working with tags and Numpy

import numpy as np
import pandas as pd

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
        transition_matrix[i, j] = transition_counts[tag_tuple]



