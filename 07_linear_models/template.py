# Author: 
# Date:
# Project: 
# Acknowledgements: 
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.

from functools import total_ordering
import numpy as np
import matplotlib.pyplot as plt

from tools import load_regression_iris
from scipy.stats import multivariate_normal


# GOAL = find the last feature in iris data with regression
# part 1.1 : scipy.stats.... does the gaussian formula
# gives out the phi matrix

# part 1.2
# goes lower because of 3 classes


# part 1.3
# outputs the weights

# part 1.4
# takes in weights, predicts the values

# part 1.5
# plot results, calculate error, the min, the max, compare...

# INDE
# Does the method work well? from above 1.5 we can see, then how can we make it better?
# add more basis functions?..


def mvn_basis(
    features: np.ndarray,
    mu: np.ndarray,
    sigma: float
) -> np.ndarray:
    '''
    Multivariate Normal Basis Function
    The function transforms (possibly many) data vectors <features>
    to an output basis function output <fi>
    Inputs:
    * features: [NxD] 150x3 is a data matrix with N D-dimensional
    data vectors.
    * mu: [MxD] 10x3 matrix of M D-dimensional mean vectors defining
    the multivariate normal distributions.
    * sigma: All normal distributions are isotropic with sigma*I covariance
    matrices (where I is the MxM 10x10 identity matrix)
    Output:
    * fi - [NxM] is the basis function vectors containing a basis function
    output fi for each data vector x in features
    '''
    N,D = features.shape
    M = mu.shape[0]
    I = np.identity(D)
    # sigma_m = sigma*I #3x3 matrix
    fi = np.zeros((N, M))

    for i in range(fi.shape[1]):
        fi[:, i] = multivariate_normal.pdf(features, mu[i,:], sigma)

    return fi


def _plot_mvn(fi: np.ndarray):

    # X,t = load_regression_iris()

    for i in range(fi.shape[1]):
        plt.plot(fi[:,i])
    
    plt.legend(['basis_1', 'basis_2', 'basis_3', 'basis_4', 'basis_5', 'basis_6', 'basis_7', 'basis_8', 'basis_9', 'basis_10'])
    plt.title("Outputs for each basis functions")
    plt.show()


def max_likelihood_linreg(
    fi: np.ndarray,
    targets: np.ndarray,
    lamda: float
) -> np.ndarray:
    '''
    Estimate the maximum likelihood values for the linear model

    Inputs :
    * Fi: [NxM] is the array of basis function vectors
    * t: [Nx1] is the target value for each input in Fi
    * lamda: The regularization constant

    Output: [Mx1], the maximum likelihood estimate of w for the linear model
    '''
    N,M = fi.shape
    I = np.identity(M)
    wml = np.linalg.inv(fi.T.dot(fi) + lamda*I).dot(fi.T.dot(targets))
    return wml


def linear_model(
    features: np.ndarray,
    mu: np.ndarray,
    sigma: float,
    w: np.ndarray
) -> np.ndarray:
    '''
    Inputs:
    * features: [NxD] is a data matrix with N D-dimensional data vectors.
    * mu: [MxD] matrix of M D dimensional mean vectors defining the
    multivariate normal distributions.
    * sigma: All normal distributions are isotropic with s*I covariance
    matrices (where I is the MxM identity matrix).
    * w: [Mx1] the weights, e.g. the output from the max_likelihood_linreg
    function.

    Output: [Nx1] The prediction for each data vector in features
    '''
    return np.matmul(mvn_basis(features, mu, sigma), w)

def _square_error(real, pred):
    sqr_error = []
    for i in range(len(real)):
        foo = real[i] - pred[i]
        error = np.power(foo, 2)
        sqr_error.append(error)
    return sqr_error

def best_parameters():
    lowest_error = 1000
    sigma = np.linspace(1,100,100)
    lamda = np.linspace(0, 1/10, 250)
    best_sigma = 0
    best_lamda = 0


    X,t = load_regression_iris()
    N,D = X.shape
    M = 10
    mu = np.zeros((M, D))
    for i in range(D):
        mmin = np.min(X[i, :])
        mmax = np.max(X[i, :])
        mu[:, i] = np.linspace(mmin, mmax, M)

    for i in range(len(sigma)):
        for j in range(len(lamda)):
            fi = mvn_basis(X, mu, sigma[i])
            wml = max_likelihood_linreg(fi, t, lamda[j])  
            prediction = linear_model(X, mu, sigma[i], wml)
            sqr_error = _square_error(t, prediction)
            total_sqr = np.sum(sqr_error)
            if total_sqr < lowest_error:
                lowest_error = total_sqr
                best_sigma = sigma[i]
                best_lamda = lamda[j]
    return lowest_error, best_sigma, best_lamda


if __name__ == '__main__':
    #  PART 1.1
    '''
    X,t = load_regression_iris()
    N,D = X.shape
    M, sigma = 10, 10
    mu = np.zeros((M, D))
    for i in range(D):
        mmin = np.min(X[i, :])
        mmax = np.max(X[i, :])
        mu[:, i] = np.linspace(mmin, mmax, M)
    fi = mvn_basis(X, mu, sigma)
    # print(fi)

    # PART 1.2
    _plot_mvn(fi)

    # PART 1.3
    lamda = 0.001
    wml = max_likelihood_linreg(fi, t, lamda)
    # print(wml)

    # PART 1.4
    prediction = linear_model(X, mu, sigma, wml)
    print(prediction)

    # PART 1.5
    # comparing predicted features to actual features

    plt.plot(range(0, len(prediction)), prediction)
    plt.plot(range(0, len(t)), t)
    plt.legend(["Predicted", "Actual"])
    plt.title("Predicted vs Actual values")
    plt.xlabel("Iterations")
    plt.ylabel("Features")
    plt.show()

    # need to find the square error and plot it to show if getting closer to real value
    sqr_error = _square_error(t, prediction)
    print(f"sum of total sqr error is: {np.sum(sqr_error)}")

    sqr_e_1 = np.sum(sqr_error[0:49])/len(sqr_error[0:49])
    sqr_e_2 = np.sum(sqr_error[50:99])/len(sqr_error[50:99])
    sqr_e_3 = np.sum(sqr_error[100:150])/len(sqr_error[100:150])
    print(sqr_e_1)
    print(sqr_e_2)
    print(sqr_e_3)

    plt.plot(range(0,len(sqr_error)), sqr_error)
    plt.title("Mean-sqr-Error")
    plt.xlabel("Iterations")
    plt.ylabel("Sqr Error")

    plt.text(10, 0.5, r'$\mu=0.03\ $')
    plt.text(60, 0.5, r'$\mu=0.07\ $')
    plt.text(115, 0.5, r'$\mu=0.08\ $')
    plt.show()
    '''
    '''
    # INDEPENDENT
    total_sqr, best_sigma, best_lamda = best_parameters()

    print(total_sqr)
    print(best_sigma)
    print(best_lamda)
    '''