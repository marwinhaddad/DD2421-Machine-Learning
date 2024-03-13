import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from labfuns import *

def mlParams(X, labels, W=None):
    assert(X.shape[0]==labels.shape[0])
    Npts,Ndims = np.shape(X)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    if W is None:
        W = np.ones((Npts,1))/float(Npts)

    mu = np.zeros((Nclasses,Ndims))
    sigma = np.zeros((Nclasses,Ndims,Ndims))
    
    # TODO: fill in the code to compute mu and sigma!
    # ==========================
    for c in classes:
        Nk = sum(labels==c)
        Xc = X[labels==c]
        mu[c] = np.mean(Xc, axis=0)
        for m in range(Ndims):
            sigma[c, m, m] = sum([(Xc[i, m] - mu[c, m])**2 for i in range(Nk)]) / Nk
    # ==========================
    return mu, sigma


X, labels = genBlobs(centers=5)
mu, sigma = mlParams(X,labels)
plotGaussian(X,labels,mu,sigma)