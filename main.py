#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 16:05:20 2017

@author: quien
"""

import numpy as np;
import numpy.random as rd;

import scipy.io as scio;
import matplotlib.pyplot as plt;
import matplotlib.image  as img;

from conv_fista import *;

a = scio.loadmat("mnist_all.mat");

D = 2;
I = 10;
X = np.zeros((D*I,28,28));
for i in range(I):
    for b in range(D):
        X[D*i+b] = -0.5+a['train'+str(b)][i,:].reshape((28,28))/255.0;

L = 20;
K = 7;
d = 10.0;
test = learner(X,L*L,K,d);

it = 0;
while it < 10:
    print "Iteration: "+str(it);
    test.step_D(5e-1,it);
    it += 1;

f,axarr = plt.subplots(L,L);
for k in range(L):
    for l in range(L):
        axarr[k,l].imshow(test.D[L*k+l,:,:],cmap='gray');
        axarr[k,l].set_xticklabels([]);
        axarr[k,l].set_yticklabels([]);
        axarr[k,l].grid(False)
plt.show()

x = a['test1'][100,:].reshape((28,28))/255.0;
A,f = test.step_A(x,0);
print f;

f,axarr = plt.subplots(L,L);
for k in range(L):
    for l in range(L):
        axarr[k,l].imshow(A[L*k+l,:,:],cmap='gray');
        axarr[k,l].set_xticklabels([]);
        axarr[k,l].set_yticklabels([]);
        axarr[k,l].grid(False)
plt.show()

s = np.zeros((x.shape[0],x.shape[1]));
f,axarr = plt.subplots(L,L);
for k in range(L):
    for l in range(L):
        s += conv2d(A[L*k+l,:,:],test.D[L*k+l,:,:],'full');
        axarr[k,l].imshow(s,cmap='gray');
        axarr[k,l].set_xticklabels([]);
        axarr[k,l].set_yticklabels([]);
        axarr[k,l].grid(False)
plt.show()

y = get_conv(A,test.D);
f,axarr = plt.subplots(1,3);
axarr[0].imshow(y,cmap='gray');
axarr[1].imshow(x,cmap='gray');
axarr[2].imshow(np.abs(x-y),cmap='gray');