import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

word_embeddings = pickle.load(open('./data/word_embeddings_subset.p', 'rb'))
len(word_embeddings)

def vec(w):
    return word_embeddings[w]

words = ['oil', 'gas', 'happy', 'sad', 'city', 'town', 'village', 'country',
        'continent', 'petroleum', 'joyful']

bag2d = np.array([vec(word) for word in words])

fig, ax = plt.subplots(figsize=(10, 10))

col1 = 3
col2 = 2

for word in bag2d:
    ax.arrow(0, 0, word[col1], word[col2], head_width=0.005, head_length=0.005, fc='r', ec='r', width = 1e-5)

ax.scatter(bag2d[:, col1], bag2d[:, col2])

for i in range(0, len(words)):
    ax.annotate(words[i], (bag2d[i, col1], bag2d[i, col2]))

plt.show()












