import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from vector_space_models.utils import get_country, compute_pca

data = pd.read_csv('./data/Files/capitals.txt', sep=' ')
data.columns = ['city1', 'country1', 'city2', 'country2']
data.head()

fh = open('./data/Files/word_embeddings_subset.p', 'rb')
word_embeddings = pickle.load(fh)
len(word_embeddings)

print("dimension: {}".format(word_embeddings['Spain'].shape[0]))
country = get_country('Athens', 'Greece', 'Cairo', word_embeddings)

# Plotting the vectors using PCA
X = np.random.rand(3, 10)
compute_pca(X, 2)

print('just for debugging')