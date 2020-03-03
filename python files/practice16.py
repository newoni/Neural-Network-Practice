# <20.3.3> by KH
'''
page 101
Global matrix factorization
'''

import numpy as np

# define a matrix that we want to

A = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print('Initial matrix')
print(A)

# Applying singular-value decomposition
# VT is already the vector we are looking in
# as the formula rturn it transposed
# while we are interested in the normal form

U, s, VT = np.linalg.svd(A)

# creting a m x n Sigma matrix
Sigma = np.zeros((A.shape[0], A.shape[1]))

# populate Sigma with n x n diagnoal matrix
Sigma[:A.shape[0], :A.shape[0]]= np.diag(s)

# select only two elements
n_elements = 2
Sigma = Sigma[:, :n_elements]
VT = VT[:n_elements, :]

# reconstruct
A_reconstructed = U.dot(Sigma.dot(VT))
print(A_reconstructed)

# Calculate the result
# By the dot product
# Between the U and sigma
# in python 3 it's possible to
# calculate the dot product using @
T = U @Sigma

# for python 2 should be
# T = U.dot(Sigma)
print('dot product between U and Sigma')
print(T)
print('dot product between A and V')
T_ = A @ VT.T
print(T_)

print('Are the dot product similar?',
      'Yes' if np.isclose(T, T_).all() else 'no')
