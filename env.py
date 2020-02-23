#!/usr/bin/env python
# coding: utf-8

import math
import numpy as np
# ### Generate the sensor environment
# generate input data streams
def gen_input(mu, mu_c, sigma, tau, L):
    '''
    Generate observation for one episode from expected mean and
    covariance matrix.

    Args:
        mu : normal mean of the observations, all zeros
        mu_c : abnormal mean of the observations
        sigma : assumed covariance matrix of the observations
        tau : the time that abnormality starts
        L : total number of observations in one episode

    Returns:
        Xn : generated observations for one episode
    '''
    # generate multi-variate gaussian
    Xn = np.random.multivariate_normal(mu, sigma, tau)
    # print(Xn.shape) # (50, 10), first dimension is exp. episode
    Xn_abnormal = np.random.multivariate_normal(mu_c, sigma, L - tau)
    # print(Xn_abnormal.shape) # (150, 10)
    Xn = np.vstack((Xn, Xn_abnormal))
    return Xn

def visualize(Xn):
    '''
    Visualize generated observations for debugging.

    Args:
        Xn : generated observations for one episode
    '''
    print(Xn.shape)
    import matplotlib.pyplot as plt
    # Plot the sampled functions
    plt.figure()
    X = np.arange(Xn.shape[0])
    for i in range(Xn.shape[1]):
        plt.plot(X, Xn[:, i], linestyle='-', marker='o', markersize=3)
    plt.xlabel('$x$', fontsize=13)
    plt.ylabel('$y = f(x)$', fontsize=13)
    plt.title('5 different function sampled from a Gaussian process')
    plt.show()


