#!/usr/bin/env python
# coding: utf-8

# In[11]:


import math
import numpy as np
import cmab
import env


# In[12]:


p = 10 # total # of sensors
q = 4 # total # of misbehaved sensors
m = 5 # total # of observed sensors

tau = 50 # error ocurrance time
L = tau + 200 # length of one exp episode
N = 10 # total experiment times

# generate sigma
# some confusion on sigma[i,i]. set it to 1.0 here. not sure.
sigma = np.full((p, p), 0.5)
np.fill_diagonal(sigma, 1.0)

cmab.gamma.lamda = 0.1 # lambda

# generate mu_c
delta = 2.0
mu = np.zeros((p, )) # normal mean
mu_c = np.zeros((p, ))
for i in range(0, int(q/2)): # first q elements to delta, error mean
    mu_c[i*2] = delta
    mu_c[i*2+1] = delta
# print(mu_c)

CMAB_ADD, CMAB_s_ADD, rdm_ADD, opt_ADD = [], [], [], []
CMAB_MAX, CMAB_s_MAX, rdm_MAX, opt_MAX = [], [], [], []
for n in range(0, N):
    Xn = env.gen_input(mu, mu_c, sigma, tau, L)
    print('mu_c', mu_c, 'iter', n)
    # env.visualize(Xn)

    # CMAB
    cmab.test.h = 6.0
    detect = cmab.CMAB(p, m, L, sigma, Xn, cmab.gamma)
    ADD = detect - tau
    print('cmab', ADD, cmab.test.max, cmab.test.h)
    if ADD > 0: CMAB_ADD.append(ADD)
    CMAB_MAX.append(cmab.test.max)
    cmab.test.max = None

    # CMAB_s
    cmab.CMAB_s.h = 1.5
    detect = cmab.CMAB_s(p, m, L, sigma, Xn, cmab.gamma)
    ADD = detect - tau
    print('cmab_s', ADD, cmab.CMAB_s.max, cmab.CMAB_s.h)
    if ADD > 0: CMAB_s_ADD.append(ADD)
    CMAB_s_MAX.append(cmab.CMAB_s.max)
    cmab.CMAB_s.max = None

    # random
    cmab.test.h = 6.0
    detect = cmab.rdm(p, m, L, sigma, Xn)
    ADD = detect - tau
    print('rdm', ADD, cmab.test.max, cmab.test.h)
    if ADD > 0: rdm_ADD.append(ADD)
    rdm_MAX.append(cmab.test.max)
    cmab.test.max = None

    # optimal, m = p
    cmab.test.h = 8.0
    detect = cmab.opt(p, L, sigma, Xn)
    ADD = detect - tau
    print('opt', ADD, cmab.test.max, cmab.test.h)
    if ADD > 0: opt_ADD.append(ADD)
    opt_MAX.append(cmab.test.max)
    cmab.test.max = None

def result(label, res_list, N):
    res_list = np.array(res_list)
    res_max = np.max(res_list)
    res_min = np.min(res_list)
    res_mean = np.mean(res_list)
    res_var = np.var(res_list)
    print('\nResult of', label)
    print('Valid percentage of samples', len(res_list) / N * 100, '%')
    print('max: {} min: {} mean: {} var: {}'.format(res_max, res_min, \
            res_mean, res_var))

#result('CMAB_ADD', CMAB_ADD, N)
#result('CMAB_MAX', CMAB_MAX, N)
#result('CMAB_s_ADD', CMAB_s_ADD, N)
#result('CMAB_s_MAX', CMAB_s_MAX, N)
#result('rdm_ADD', rdm_ADD, N)
#result('rdm_MAX', rdm_MAX, N)
result('opt_ADD', opt_ADD, N)
result('opt_MAX', opt_MAX, N)
