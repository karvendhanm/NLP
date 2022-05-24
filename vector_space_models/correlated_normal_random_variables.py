import numpy as np                         # Linear algebra library
import matplotlib.pyplot as plt            # library for visualization
from sklearn.decomposition import PCA      # PCA library
import pandas as pd                        # Data frame library
import math                                # Library for math functions
import random

from scipy.stats import pearsonr

np.random.seed(100)

std1 = 1
std2 = 0.333

x = np.random.normal(0, std1, 1000)
y = np.random.normal(0, std2, 1000)

row_before_centering = pearsonr(x, y)

# centering the data
x = x - np.mean(x)
y = y - np.mean(y)

row_after_centering = pearsonr(x, y)

n = 1
angle = np.arctan(1/n)
angle_in_degrees = (angle * 180)/math.pi

rotationMatrix = np.array([
    [np.cos(angle), np.sin(angle)],
    [-np.sin(angle), np.cos(angle)]
])

xy = np.concatenate(([x] , [y]), axis=0).T
data = np.dot(xy, rotationMatrix)

data = pd.DataFrame(data, columns=['x', 'y'])
row_after_rotation = pearsonr(data.x, data.y)

plt.scatter(data.x, data.y)
plt.show()

# Let us print the original and the resulting transformed system using the result of the PCA in the same plot
# alongside with the 2 Principal Component vectors in red and blue

plt.scatter(data.x, data.y)

# Apply PCA. In theory, Eigen matrix must be the inverse of the original rotation matrix.
pca = PCA(n_components=2)

# Create the transformational model for this data. Internally it gets the
# rotational matrix and the explained variance.
pcaTr = pca.fit(data)

# rotational matrix or eigen vectors
pcaTr.components_

# explained variance or eigen values
pcaTr.explained_variance_

dataPCA = pcaTr.transform(data)
dataPCA = pd.DataFrame(dataPCA, columns = ['x', 'y'])

plt.scatter(dataPCA.x, dataPCA.y)

plt.plot([0, rotationMatrix[0][0] * std1 * 3], [0, rotationMatrix[0][1] * std1 * 3], 'k-', color='red')
plt.plot([0, rotationMatrix[1][0] * std2 * 3], [0, rotationMatrix[1][1] * std2 * 3],  'k-', color='green')

plt.show()



