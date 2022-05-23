import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import math
import random

from scipy.stats import pearsonr

np.random.seed(1)
n = 1 # The amount of correlation
x = np.random.uniform(1, 2, 1000) # Generate 1000 samples from uniform random variable
y = x.copy() * n

# centering the variables x and y
x = x - np.mean(x)
y = y - np.mean(y)

_dict = {
    'x': x,
    'y': y
}

data = pd.DataFrame(_dict)
pca = PCA(n_components=2)
plt.scatter(data.x, data.y)

pcaTr = pca.fit(data)
rotatedData = pcaTr.transform(data)

pcaData = pd.DataFrame(rotatedData, columns=['PCA1', 'PCA2'])
plt.scatter(pcaData.PCA1, pcaData.PCA2)
plt.show()











