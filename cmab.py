#!/usr/bin/env python
# coding: utf-8

# In[99]:


import math
import numpy as np
from itertools import combinations
import random

# ### Update Vn and Mun

# In[100]:


def gen_E_zn(zn, m, p):
    # assert check
    assert(zn.shape == (m, ))
    E_zn = np.zeros((m, p))
    row = 0
    for z in zn:
        E_zn[row, z] = 1.0
        row += 1
    # print("E_zn: {}".format(E_zn))
    return E_zn

def update_vn(vn_old, zn, sigma, lamda, m, p):
    '''
    Update estimated variance for each sensor variable
    based on previous observations.
    '''
    # assert check
    assert(vn_old.shape == (p, p))
    assert(zn.shape == (m, ))
    assert(sigma.shape == (p, p))

    # generate E_zn
    E_zn = gen_E_zn(zn, m, p)

    # generate sigma_zn
    sigma_zn = sigma[zn, :][:, zn]
    sigma_zn_inv = np.linalg.inv(sigma_zn)

    # calculate vn update
    tmp = np.matmul(E_zn.T, sigma_zn_inv)
    tmp = np.matmul(tmp, E_zn)
    # print("tmp: {}".format(tmp))
    vn_new_inv = (1 - lamda) * np.linalg.inv(vn_old) + tmp
    return np.linalg.inv(vn_new_inv)

def update_mun(mun_old, zn, sigma, lamda, m, p, Xn, vn):
    '''
    Update estimated mean for each sensor variable
    based on previous observations.
    '''
    # assert check
    assert(mun_old.shape == (p, ))
    assert(zn.shape == (m, ))
    assert(sigma.shape == (p, p))
    assert(Xn.shape == (m, ))
    assert(vn.shape == (p, p))

    # generate E_zn
    E_zn = gen_E_zn(zn, m, p)

    # generate sigma_zn
    sigma_zn = sigma[zn, :][:, zn]
    sigma_zn_inv = np.linalg.inv(sigma_zn)

    # calculate mun update
    tmp = np.matmul(E_zn.T, sigma_zn_inv)
    tmp = np.matmul(tmp, Xn)
    # print("tmp: {}".format(tmp))
    mun_new = np.matmul(vn, (1 - lamda) * mun_old + tmp)
    return mun_new


# ### Test for bound h

# In[101]:


def test(mu, v):
    '''
    Judging function of whether abnormality happens.
    Calculate detection power and compare it with the threshold.

    Args:
        mu : current estimated mean of each sensor variable
        v : current estimated variance of each sensor variable

    Attributes:
        test.max : current maximum detection power
        test.h : threshold of detection power

    Returns:
        True : if an abnormality is detected
        False : no abnormality is detected
    '''
    # The setting of test.h should makes ARL = 200
    # when returning true, the abnormal is detected.
    # Otherwise, it is not detected.
    # test.h should be set before running test.
    assert(test.h != None)
    v_inv = np.linalg.inv(v)
    t = np.matmul(mu.T, v_inv)
    t = np.matmul(t, mu)
    # print("test value: {} max value: {}".format(t, test.max))
    if test.max is None:
        test.max = t
    elif t > test.max:
        test.max = t

    if t > test.h:
        return True
    return False

test.max = None
test.h = None


# ### Candidate gamma functions

# In[102]:


def gamma(n):
    '''
    Gamma function to balance exploitation and exploration.
    '''
    tmp = 1.0 - math.pow((1.0 - gamma.lamda), n)
    tmp = tmp / (gamma.lamda * gamma.lamda)
    return 2.0 * math.log(tmp)

gamma.lamda = None

def gamma_zero(n):
    '''
    Constant gamma function.
    Greedy exploitation, no exploration.
    '''
    return 0.0


# ### CMAB for adaptively sampling Zt

# In[103]:


def CMAB(p, m, L, sigma, Xn, gamma_func):
    '''
    Apply CMAB for adaptive sampling for one episode.

    Args:
        p : total # of sensors
        m : total # of observed sensors
        L : length of one experiment episode
        sigma : assumed covariance matrix of the observations
        Xn : oberservations, (observation #) * (sensor #)
        gamma_func : callable to compute gamma given observation #

    Returns:
        detect : the number of observation that detects abnormality
            in this episode. If no abnormality is detected, return L.
    '''
    # assert check
    assert(Xn.shape == (L, p))

    # init mu and v, not sure
    mun_old = np.zeros((p, ))
    vn_old = np.zeros((p, p))
    np.fill_diagonal(vn_old, 1.0)

    # observe all sensors for once
    for iter in range(0, math.ceil(p/m)):
        zn = np.arange(iter * m, (iter + 1) * m)
        # print("zn: {}".format(zn))
        Xn_observe = Xn[iter, zn]
        # print("Xn_observe: {}".format(Xn_observe))

        # update vn and mun
        vn_new = update_vn(vn_old, zn, sigma, gamma.lamda, m, p)
        # print("vn_old: {}".format(vn_old))
        # print("vn_new: {}".format(vn_new))
        mun_new = update_mun(mun_old, zn, sigma, gamma.lamda, m, p, Xn_observe, vn_new)
        # print("mun_old: {}".format(mun_old))
        # print("mun_new: {}".format(mun_new))

        # copy new to old
        vn_old = np.copy(vn_new)
        mun_old = np.copy(mun_new)

    # continue experiments and get ADD
    detect = L # init to the total # of observation
    for iter in range(math.ceil(p/m), L):
        gamma_n = gamma_func(iter)
        # print(gamma_n)
        comb = combinations(range(0, p), m)
        max_r, max_c = None, None

        # iterate all possible combinations
        for c in comb:
            sigma_zk = sigma[c, :][:, c]
            phi_zk = np.linalg.inv(sigma_zk)
            cur_r = 0.0
            # now we calculate the upper confidence bound of the estimated
            # reward of choosing arm Zk at epoch
            for i in range(m):
                for j in range(m):
                    phi_zk_tmp = phi_zk[i, j]
                    phi_zk_wgamma = phi_zk_tmp * phi_zk_tmp * gamma_n
                    cur_r += phi_zk_tmp * mun_old[c[i]] * mun_old[c[j]]
                    cur_r += math.sqrt(phi_zk_wgamma * mun_old[c[i]] * \
                            mun_old[c[i]] * vn_old[c[j], c[j]])
                    cur_r += math.sqrt(phi_zk_wgamma * mun_old[c[j]] * \
                            mun_old[c[j]] * vn_old[c[i], c[i]])
                    cur_r += gamma_n * math.sqrt(phi_zk_tmp * phi_zk_tmp * \
                            vn_old[c[i], c[i]] * vn_old[c[j], c[j]])
            # print("cur_r {} cur_c {} max_r {} max_c {}".format(cur_r, c, max_r, max_c))
            if max_r is None:
                max_r = cur_r
                max_c = c
            elif cur_r > max_r:
                max_r = cur_r
                max_c = c

        # print("iter {} selected sensors: {}".format(iter, max_c))
        zn = np.array(max_c)
        Xn_observe = Xn[iter, zn]

        # based on the selected arm, update mun and vn
        vn_new = update_vn(vn_old, zn, sigma, gamma.lamda, m, p)
        mun_new = update_mun(mun_old, zn, sigma, gamma.lamda, m, p, Xn_observe, vn_new)

        # copy new to old
        vn_old = np.copy(vn_new)
        mun_old = np.copy(mun_new)

        if test(mun_old, vn_old) == True:
            # print("Abnormal detected at iter {}".format(iter))
            detect = iter
            break

    return detect


# ### CMAB(s) algorithm

# In[104]:


def CMAB_s(p, m, L, sigma, Xn, gamma_func):
    '''
    Apply CMAB_s for adaptive sampling for one episode.
    Simple version of CMAB.

    Args:
        p : total # of sensors
        m : total # of observed sensors
        L : length of one experiment episode
        sigma : assumed covariance matrix of the observations
        Xn : oberservations, (observation #) * (sensor #)
        gamma_func : callable to compute gamma given observation #

    Returns:
        detect : the number of observation that detects abnormality
            in this episode. If no abnormality is detected, return L.
    '''
    # assert check
    assert(Xn.shape == (L, p))

    # init mu and v, not sure
    mun_old = np.zeros((p, ))
    vn_old = np.zeros((p, p))
    np.fill_diagonal(vn_old, 1.0)

    # observe all sensors for once
    for iter in range(0, math.ceil(p/m)):
        zn = np.arange(iter * m, (iter + 1) * m)
        # print("zn: {}".format(zn))
        Xn_observe = Xn[iter, zn]
        # print("Xn_observe: {}".format(Xn_observe))

        # update vn and mun
        vn_new = update_vn(vn_old, zn, sigma, gamma.lamda, m, p)
        # print("vn_old: {}".format(vn_old))
        # print("vn_new: {}".format(vn_new))
        mun_new = update_mun(mun_old, zn, sigma, gamma.lamda, m, p, Xn_observe, vn_new)
        # print("mun_old: {}".format(mun_old))
        # print("mun_new: {}".format(mun_new))

        # copy new to old
        vn_old = np.copy(vn_new)
        mun_old = np.copy(mun_new)

    # continue experiments and get ADD

    detect = L # init to the total # of observation
    for iter in range(math.ceil(p/m), L):
        gamma_n = gamma_func(iter)
        # print(gamma_n)

        # Greedily select the sensors that have highest rewards
        r = np.absolute(mun_old) + np.sqrt(gamma_n * np.diag(vn_old))
        zn = np.argsort(r)[p-m:]
        Xn_observe = Xn[iter, zn]

        # based on the selected arm, update mun and vn
        vn_new = update_vn(vn_old, zn, sigma, gamma.lamda, m, p)
        mun_new = update_mun(mun_old, zn, sigma, gamma.lamda, m, p, Xn_observe, vn_new)

        # copy new to old
        vn_old = np.copy(vn_new)
        mun_old = np.copy(mun_new)

        # note the detection here is also different from CMAB
        t = np.dot(mun_old, mun_old)
        if CMAB_s.max == None:
            CMAB_s.max = t
        elif t > CMAB_s.max:
            CMAB_s.max = t

        if t > CMAB_s.h:
            # print("Abnormal detected at iter {}".format(iter))
            detect = iter
            break

    return detect

CMAB_s.max = None
CMAB_s.h = None


# ### random algorithm

def rdm(p, m, L, sigma, Xn):
    '''
    Random selection.

    Args:
        p : total # of sensors
        m : total # of observed sensors
        L : length of one experiment episode
        sigma : assumed covariance matrix of the observations
        Xn : oberservations, (observation #) * (sensor #)

    Returns:
        detect : the number of observation that detects abnormality
            in this episode. If no abnormality is detected, return L.
    '''
    # assert check
    assert(Xn.shape == (L, p))

    # init mu and v, not sure
    mun_old = np.zeros((p, ))
    vn_old = np.zeros((p, p))
    np.fill_diagonal(vn_old, 1.0)

    # continue experiments and get ADD
    detect = L # init to the total # of observation
    for iter in range(0, L):
        rdm_c = random.sample(range(0, p), m)
        zn = np.array(rdm_c)
        Xn_observe = Xn[iter, zn]

        # based on the selected arm, update mun and vn
        vn_new = update_vn(vn_old, zn, sigma, gamma.lamda, m, p)
        mun_new = update_mun(mun_old, zn, sigma, gamma.lamda, m, p, Xn_observe, vn_new)

        # copy new to old
        vn_old = np.copy(vn_new)
        mun_old = np.copy(mun_new)

        if test(mun_old, vn_old) == True:
            # print("Abnormal detected at iter {}".format(iter))
            detect = iter
            break

    return detect

# ### optimal algorithm: all observable

def opt(p, L, sigma, Xn):
    '''
    Optimal algorithm.

    Args:
        p : total # of sensors
        m : total # of observed sensors
        L : length of one experiment episode
        sigma : assumed covariance matrix of the observations
        Xn : oberservations, (observation #) * (sensor #)

    Returns:
        detect : the number of observation that detects abnormality
            in this episode. If no abnormality is detected, return L.
    '''
    # assert check
    assert(Xn.shape == (L, p))

    # init mu and v, not sure
    mun_old = np.zeros((p, ))
    vn_old = np.zeros((p, p))
    np.fill_diagonal(vn_old, 1.0)

    # continue experiments and get ADD
    detect = L # init to the total # of observation
    for iter in range(0, L):
        zn = np.array(range(0, p))
        Xn_observe = Xn[iter, :]

        # based on the selected arm, update mun and vn
        vn_new = update_vn(vn_old, zn, sigma, gamma.lamda, p, p)
        mun_new = update_mun(mun_old, zn, sigma, gamma.lamda, p, p, Xn_observe, vn_new)

        # copy new to old
        vn_old = np.copy(vn_new)
        mun_old = np.copy(mun_new)

        if test(mun_old, vn_old) == True:
            # print("Abnormal detected at iter {}".format(iter))
            detect = iter
            break

    return detect
