# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 11:50:01 2015

@author: anaiman
"""

import numpy as np
import scipy.optimize as opt
from matplotlib import pyplot as plt

def linear_regression(theta, X, y):
    
    """
    Arguments:
    theta - vector containing parameter values to optimize
    X - examples (in an array)
        X(i,j) is the i'th coordinate of the j'th example
    y - target value for each example
        y(j) is the target for example j
    
    Returns:
    f - least squares objective function value for given parameters
    g - gradient of objective function value for given parameters
    """
    
    [n, m] = np.shape(X)
    
    f = 0
    g = np.zeros(np.shape(theta))
    
    h = np.transpose(theta).dot(X)
    dh = h - y
    
    f = 0.5 * np.transpose(dh).dot(dh)
    g = X.dot(dh)
    
    return f, g


# Prepare data

# Load housing data from file
data = np.loadtxt('housing.data')
# Shuffle examples into random order
np.random.shuffle(data)
# Put examples into columns
data = np.transpose(data)
# Add a row of ones to allow an intercept feature
data = np.vstack((np.ones((1, data.shape[1])), data))

# Split data into training and test sets
# Last row is the median home price
n_train = 400
train_x = data[0:-1, 0:n_train]
train_y = data[-1, 0:n_train]
[n, m] = np.shape(train_x)

test_x = data[0:-1, n_train+1:-1]
test_y = data[-1, n_train+1:-1]

# Initialize the coefficient vector to random values
theta0 = np.random.rand(n,1)

# Minimize the linear regression objective function
result = opt.minimize(linear_regression, theta0, args=(train_x, train_y), 
                      jac=True, options={'maxiter': 200, 'disp': True})
theta = result.x

# How'd we do?
train_predicted = np.transpose(theta).dot(train_x)
s = train_predicted - train_y
train_rms = np.sqrt(np.mean(s*s))
print "RMS Training Error: " + str(train_rms)

test_predicted = np.transpose(theta).dot(test_x)
s = test_predicted - test_y
test_rms = np.sqrt(np.mean(s*s))
print "RMS Testing Error: " + str(test_rms)

#sorted_indices = np.argsort(train_y)
#plt.plot(train_predicted[sorted_indices], 'bx')
#plt.plot(train_y[sorted_indices], 'rx')
#plt.show()

sorted_indices = np.argsort(test_y)
plt.plot(test_predicted[sorted_indices], 'bx', label='Predicted Price')
plt.plot(test_y[sorted_indices], 'rx', label='Actual Price')
plt.legend(loc='upper left')
plt.xlabel('House #')
plt.ylabel('House price ($1000s)')
plt.show()
