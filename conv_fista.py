#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 15:49:26 2017

@author: quien
"""

import numpy as np;
import numpy.random as rd;

import scipy.io as scio;
from scipy.signal import convolve2d as conv2d;
from scipy.signal import correlate2d as corr2d;

def get_conv(A,B):
    C = np.zeros((A.shape[1]+B.shape[1]-1,A.shape[2]+B.shape[2]-1));
    for l in range(A.shape[0]):
        C += conv2d(A[l],B[l],'full');
    return C;

def get_corr(A,B):
    C = np.zeros((B.shape[0],A.shape[0]-B.shape[1]+1,A.shape[1]-B.shape[2]+1));
    for l in range(C.shape[0]):
        C[l] = corr2d(A,B[l],'valid');
    return C;

def J(X,D,A):
    Z = get_conv(A,D)-X;
    return 0.5*np.mean(np.power(Z,2));

def F(X,D,A,lam):
    return J(X,D,A)+lam*np.mean(np.abs(A));

def P(X,D,A,lam,dt):
    Y = A-dt*get_corr(get_conv(A,D)-X,D)/X.size;
    Z = np.abs(Y)-dt*lam/Y.size;
    Z = Z * (Z > 0);
    return Z*np.sign(Y);

class learner:
    def __init__(self,X,L,K,lam):
        self.X = X;
        
        self.L = L;
        self.K = K;
        
        self.lam = lam;
        
        self.D  = rd.randn(L,K,K);
        for l in range(self.L):
            self.D[l] /= np.maximum(1.0,np.linalg.norm(self.D[l]));

        self.mD = np.zeros((L,K,K));
        self.nD = np.zeros((L,K,K));

        self.it = 0;
        self.IT = 10;
    
    
    def step_A(self,x,it):
        A = get_corr(x,self.D);
        A += 0.05*rd.randn(A.shape[0],A.shape[1],A.shape[2])/A.size;
        Al = A;
        A_ = A;
        t = 0;
        eta = 0.0;
        c = 0.5;
        k = 0.5;
        a0 = 1.0;
        while t < 200:
            Al = A;
            
            
            dJdA = get_corr(get_conv(A_,self.D)-x,self.D)/x.size
            m = c*np.sum(dJdA*(dJdA+self.lam*np.sign(A_)/A_.size));
            dt = a0;
            FA_ = F(x,self.D,A_,self.lam);

            i = 0;
            while i < 10:
                i += 1;
                if FA_ >= F(x,self.D,P(x,self.D,A_,self.lam,dt),self.lam)+dt*m:
                    break;
                dt *= k;
            A = P(x,self.D,A_,self.lam,dt);
            f = F(x,self.D,A,self.lam);
            print it,t,f,dt;

            eta_n = (1.0+np.sqrt(1.0 + 4.0*eta*eta))/2.0;
            mu = (1.0-eta)/eta_n;
            A_ = (1.0-mu)*A+mu*Al;
            eta = eta_n;
            t += 1;
        
        f = F(x,self.D,A,self.lam);
        return A,f;
    
    def step_D(self,dt,iit):
        dD = np.zeros(self.D.shape);
        for i in range(self.X.shape[0]):
            x = self.X[i];
            A,f = self.step_A(x,(iit,i));
            print iit,i,f;
            dD = get_corr(get_conv(A,self.D)-x,A)/(x.size*self.IT);
            self.D -= dt*dD;
            for l in range(self.L):
                self.D[l] /= np.maximum(1.0,np.linalg.norm(self.D[l]));
            #self.it += 1;
        