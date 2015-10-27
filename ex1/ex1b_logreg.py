# -*- coding: utf-8 -*-
"""
Python script implementing the second (logistic regression) exercise

Created on Tue Oct 27 10:48:29 2015

@author: anaiman
"""

import numpy as np
import scipy.optimize as opt
from matplotlib import pyplot as plt

def loadMNISTImages(filename):

    with open(filename, 'rb') as f:
        # Header
        dt = np.dtype('>i4')
        magic = np.fromfile(f, dtype=dt, count=1)
        numImages = np.fromfile(f, dtype=dt, count=1)
        numRows = np.fromfile(f, dtype=dt, count=1)
        numCols = np.fromfile(f, dtype=dt, count=1)
        
        # Data
        dt = np.dtype('B1')
        images = np.fromfile(f, dtype=dt, count=-1)
        #first = images[0:28*28].reshape(28,28)
        
        # The following maniuplations don't seem to be necessary here
        #images = np.reshape(images, (numCols, numRows, numImages))
        #images = np.transpose(images, (1,0,2))
    
    images = np.reshape(images, (numCols*numRows, numImages), 'F')
    images = images/255.0
    
    #first = images[0:28*28,0].reshape(28,28)
    #plt.imshow(first, 'gray')
    
    return images
    
def loadMNISTLabels(filename):
    
    with open(filename, 'rb') as f:
        # Header
        dt = np.dtype('>i4')
        magic = np.fromfile(f, dtype=dt, count=1)
        numLabels = np.fromfile(f, dtype=dt, count=1)
        
        # Data
        dt = np.dtype('B1')
        labels = np.fromfile(f, dtype=dt, count=-1)
    
    #print str(labels[0])
    return labels
    
def logistic_regression(theta, X, y):
    
    """
    Arguments:
    theta - vector containing parameter values to optimize
    X - examples (in an array)
        X(i,j) is the i'th coordinate of the j'th example
    y - target value for each example
        y(j) is the target for example j
    
    Returns:
    f - logistic objective function value for given parameters
    g - gradient of objective function value for given parameters
    """
    
    h = 1.0 / (1 + np.exp(-1.0*theta.dot(X)))
    
    f = -1.0*(y.dot(np.log(h)) + (1-y).dot(np.log(1-h)))
    g = X.dot(h - y)
    
    return f, g

def main():
    
    # Prepare training data
    X = loadMNISTImages('train-images-idx3-ubyte')
    y = loadMNISTLabels('train-labels-idx1-ubyte')
    
    # Binary digits only
    mask = np.any([y == 0, y == 1], axis=0)
    X = X[:, mask]
    y = y[mask]
    
    # Shuffle data into random order
    indices = np.random.permutation(y.shape[0])
    y = y[indices]
    X = X[:, indices]
    
    # Normalize data per pixel - get mean ~0 and std ~1
    s = np.std(X, axis=1)
    m = np.mean(X, axis=1)
    X = (X.T-m).T
    X = (X.T/(s+0.1)).T
    
    train_X = X
    train_y = y
    
    # Prepare test data
    
    X = loadMNISTImages('t10k-images-idx3-ubyte')
    y = loadMNISTLabels('t10k-labels-idx1-ubyte')
    
    # Binary digits only
    mask = np.any([y == 0, y == 1], axis=0)
    X = X[:, mask]
    y = y[mask]
    
    # Shuffle data into random order
    indices = np.random.permutation(y.shape[0])
    y = y[indices]
    X = X[:, indices]
    
    # Normalize data using same mean and scale as training data
    X = (X.T-m).T
    X = (X.T/(s+0.1)).T
    
    test_X = X
    test_y = y
    
    # Add a row of ones to allow an intercept feature
    train_X = np.vstack((np.ones((1, train_X.shape[1])), train_X))
    test_X = np.vstack((np.ones((1, test_X.shape[1])), test_X))
    
    [n, m] = np.shape(train_X)
    
    # Initialize the coefficient vector to random values
    theta0 = np.random.rand(n,1)*0.001
    
    # Minimize the linear regression objective function
    result = opt.minimize(logistic_regression, theta0, args=(train_X, train_y), 
                          jac=True, options={'maxiter': 100, 'disp': True})
    theta = result.x
    
    # How'd we do?
    predicted = [1.0 / (1 + np.exp(-1.0*theta.dot(train_X))) > 0.5]
    actual = [train_y > 0.5]
    correct = (predicted[0] == actual[0]).sum()
    accuracy = correct / float(train_y.shape[0])
    print "Training Accuracy: " + str(accuracy)
    
#    wrong = np.where(predicted[0] != actual[0])
#    first = train_X[1:,wrong[0][0]].reshape(28,28)
#    plt.imshow(first, 'gray')
    
    predicted = [1.0 / (1 + np.exp(-1.0*theta.dot(test_X))) > 0.5]
    actual = [test_y > 0.5]
    correct = (predicted[0] == actual[0]).sum()
    accuracy = correct / float(test_y.shape[0])
    print "Test Accuracy: " + str(accuracy)
    
    wrong = np.where(predicted[0] != actual[0])
    first = test_X[1:,wrong[0][0]].reshape(28,28)
    plt.imshow(first, 'gray')

if __name__ == '__main__':
    main()
