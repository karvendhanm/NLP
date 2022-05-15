import numpy as np

alist = [1, 2, 3, 4, 5]
narray = np.array([1, 2, 3, 4])

print(alist)
print(narray)

print(type(alist))
print(type(narray))

print(narray + narray)
print(alist + alist)

print(narray * 3)
print(alist * 3)

# Matrix or Array of Arrays
npmatrix1 = np.array([narray, narray, narray])
npmatrix2 = np.array([alist, alist, alist])
npmatrix3 = np.array([narray, [1, 1, 1, 1], narray])

print(npmatrix1)
print(npmatrix2)
print(npmatrix3)

okmatrix = np.array([[1, 2], [3, 4]])
print(okmatrix)
print(okmatrix.shape)
print(okmatrix * 2)

badmatrix = np.array([[1, 2], [3, 4], [5, 6, 7]])
print(badmatrix)
print(badmatrix * 2)

# Scaling and translating matrices:
print(okmatrix)
print((okmatrix * 2) + 1)

# Adding two compatible matrices
result1 = okmatrix + okmatrix
print(result1)

# Subtracting two compatible matrices. this is called the difference vector
result2 = okmatrix - okmatrix
print(result2)

# The product operator * when used on arrays or matrices indicates element-wise multiplications.
# Do not confuse it with the dot product.
result = okmatrix * okmatrix
print(result)

#Transpose a matrix:
matrix3x2 = np.array([[1, 2], [3, 4], [5, 6]])
print(matrix3x2)
print(matrix3x2.T)

print(narray)
print(narray.T)

anotherarray = np.array([[1, 2, 3, 4]])
print(anotherarray)
print(anotherarray.T)

# Get the norm of a nparray or matrix
nparray1 = np.array([1, 2, 3, 4])
np.linalg.norm(nparray1)

nparray2 = np.array([[1, 2], [3, 4]]) # Define a 2 x 2 matrix. Note the 2 level of square brackets
norm2 = np.linalg.norm(nparray2)
print(norm2)

# axis of the norm function
nparray3 = np.array([[1, 1], [2, 2], [3, 3]])
np.linalg.norm(nparray3, axis = 1)
np.linalg.norm(nparray3, axis = 0)

# The dot product between arrays
narray4 = np.array([[2, 3], [4, 5]])
np.dot(narray4, narray4)
np.dot(narray4, narray4.T)

nparray1 = np.array([1, 2, 3, 4])
nparray2 = np.array([5, 6, 7, 8])

nparray1.shape

flavor1 = np.dot(nparray1, nparray2)
print(flavor1)

flavour2 = np.sum(nparray1 * nparray2)
print(flavour2)

flavour3 = nparray1 @ nparray2
print(flavour3)

# Sums by rows and columns
nparray2 = np.array([[1, -1], [2, -2], [3, -3]])
nparray2.shape

np.sum(nparray2, axis = 0)
np.sum(nparray2, axis = 1)

# Get the mean by rows or columns
nparray2 = np.array([[1, -1], [2, -2], [3, -3]])
np.mean(nparray2)
np.mean(nparray2, axis=0)
np.mean(nparray2, axis=1)

# Center the columns of a matrix
nparray2 = np.array([[1, -1], [2, -2], [3, -3]])
nparrayCentered = nparray2 - np.mean(nparray2, axis=0)
print(nparrayCentered)
np.mean(nparrayCentered, axis=0)

# row centering
nparray2 = np.array([[1, 3], [2, 4], [3, 5]])
row_mean = np.mean(nparray2, axis=1)
print(row_mean)
nparrayRowCentered = (nparray2.T - np.mean(nparray2, axis=1)).T
np.mean(nparrayRowCentered, axis=1)























