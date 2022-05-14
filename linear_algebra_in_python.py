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

#