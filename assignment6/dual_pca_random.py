import numpy as np
import matplotlib.pyplot as plt

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


N = 10
d = 2

points = np.random.multivariate_normal(np.array([0,0]), np.array([[1,1.5],[0.2, 2]]), N)

U, S, VT, mu, C = pca(points.T)
print("U", U)

U2, S2, VT, mu, C = pca2N(points.T)
U2 = U2[:d,:d].T
S = S[:d]
print("U2", U2)

plt.clf()
for i, pt in enumerate(points):
    plt.plot(pt[0], pt[1], 'r+')
    plt.text(pt[0], pt[1], str(i))

mu = np.mean(points, axis=0)

plt.plot(mu[0],mu[1],'b+')

plt.plot([mu[0], mu[0]+S[0]*U[0,0]],[mu[1], mu[1]+S[0]*U[0,1]],'r')
plt.plot([mu[0], mu[0]+S[1]*U[1,0]],[mu[1], mu[1]+S[1]*U[1,1]],'g')

plt.plot([mu[0], mu[0]+S2[0]*U2[0,0]],[mu[1], mu[1]+S2[0]*U2[0,1]],'c')
plt.plot([mu[0], mu[0]+S2[1]*U2[1,0]],[mu[1], mu[1]+S2[1]*U2[1,1]],'b')

plt.gca().axis('equal')

plt.draw()
plt.pause(0.01)
plt.waitforbuttonpress()

