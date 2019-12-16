import numpy as np
"""
    SVD and PCA from scratch
"""
# https://machinelearningmastery.com/singular-value-decomposition-for-machine-learning/
# https://machinelearningmastery.com/calculate-principal-component-analysis-scratch-python/
# https://towardsdatascience.com/my-notes-for-singular-value-decomposition-with-interactive-code-feat-peter-mills-7584f4f2930a

## SVD ##
# also known as matrix factorization; try to describe a matrix using its constiuent components
# A = U . Sigma . V^T
# A - Matrix (mxn) U (mxm) Sigma (mxn) V^T (nxn)
# diagonal values in Sigma are the singular values
# columns of U are the left-singular vectors; columns of V are the right-singular vectors

#m = 20
#n = 10
#A = np.randn.norm((m, n))
A = np.matrix([[3, 1, 1], [-1, 3, 1]])
print('Starting matrix A \n', A)
print(A.shape)

# In order to find U, start with A*A^T
tmpU = np.matmul(A, A.T) # which is basiscally A*A.T
print(tmpU)

# next step: find eigenvalues and -vectors of tmpMM
tmpU_eigVals, tmpU_eigVecs = np.linalg.eig(tmpU)
print('Eigenvalues and Eigenvectors:\n', tmpU_eigVals,'\n', tmpU_eigVecs)

# V is similar to finding U
tmpV = np.matmul(A.T, A)
print(tmpV)

tmpV_eigVals, tmpV_eigVecs = np.linalg.eig(tmpV)
print('Eigenvalues and Eigenvectors:\n', tmpV_eigVals,'\n', tmpV_eigVecs)