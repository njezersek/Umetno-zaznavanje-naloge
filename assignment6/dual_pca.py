import numpy as np

def pca2(X):
    m, N = X.shape

    # calculate the mean value
    mu = np.mean(X, axis=1, keepdims=True)
    
    # center the data
    X_d =  X - mu
    
    # compute the dual covariance matrix
    C_ = 1/(m-1) * (X_d.T @ X_d)
    
    # SVD of dual covariance matrix
    U, S, VT = np.linalg.svd(C_)
    
    U = X_d @ U @ np.diag(np.sqrt(1/(S*(m-1))))

    return U, S, VT, mu, C_

def pca2N(X):
    m, N = X.shape

    # calculate the mean value
    mu = np.mean(X, axis=1, keepdims=True)
    
    # center the data
    X_d =  X - mu
    
    # compute the dual covariance matrix
    C_ = 1/(N-1) * (X_d.T @ X_d)
    
    # SVD of dual covariance matrix
    U, S, VT = np.linalg.svd(C_)
    
    U = X_d @ U @ np.diag(np.sqrt(1/(S*(N-1))))

    return U, S, VT, mu, C_

def pca(X):
    m, N = X.shape
    # calculate the mean value
    mu = 1/N * np.sum(X, axis=1, keepdims=True)
    
    # center the data
    X_d =  X - mu
    
    # compute the covariance matrix
    C = 1/(N-1) * X_d @ X_d.T
    
    U, S, VT = np.linalg.svd(C)
    
    return U, S, VT, mu, C


points = np.array([[1., 6., 5., 1., 0.],
       				[0., 2., 4., 3., 1.]])

U, S, VT, mu, C = pca(points)
print("PCA")
print("Lastni vektorji:")
print(U[:,:2])
print("Lastne vrednosti:")
print(S[:2])
print("")


U, S, VT, mu, C = pca2(points)

print("Dual PCA")
print("Lastni vektorji:")
print(U[:,:2])
print("Lastne vrednosti:")
print(S[:2])
print("")

U, S, VT, mu, C = pca2N(points)

print("Dual PCA z (N-1) namesto (m-1)")
print("Lastni vektorji:")
print(U[:,:2])
print("Lastne vrednosti:")
print(S[:2])
print("")
