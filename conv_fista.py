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

def dJdA(X,D,A):
    return get_corr(get_conv(A,D)-X,D)/X.size;

def dJdD(X,D,A):
    return get_corr(get_conv(A,D)-X,A)/X.size;

def F(X,D,A,lam):
    return J(X,D,A)+lam*np.mean(np.abs(A));

def prox(Y,eta):
    Z = np.abs(Y)-eta/Y.size;
    Z = Z * (Z > 0);
    return Z*np.sign(Y);

def root(lt,the,t):
    if lt == 0.0:
        return 1.0;
    else:
        tlt = t / lt;
        det = np.sqrt(tlt*the*the+0.25*tlt*the*the*the*the);
        return max(-0.5*tlt*the*the+det,-0.5*tlt*the*the-det);
class learner:
    def __init__(self,X,L,K,lam):
        self.X = X;
        
        self.L = L;
        self.K = K;
        
        self.lam = lam;
        
        self.D  = np.ones((L,K,K))+0.1*rd.randn(L,K,K);
        for l in range(self.L):
            self.D[l] -= np.mean(self.D[l]);
            self.D[l] /= np.linalg.norm(self.D[l])+1e-1;
        self.dD = np.zeros(self.D.shape);

        self.it = 0;
        self.IT = 5;
    
    
    def step_A(self,x,it):
        X = get_corr(x,self.D);
        V = np.copy(X);
        X += 0.001*rd.randn(X.shape[0],X.shape[1],X.shape[2]);
        V += 0.001*rd.randn(X.shape[0],X.shape[1],X.shape[2]);

        t = 0;
        
        dt = 100.0;
        the =  0.5;
        while t < 50:
            
            
            #line search
            d = 10.0;
            the_n = root(dt,the,d);
            
            Y = (1-the_n)*X+the_n*V;
            jY = J(x,self.D,Y);
            dJdY = dJdA(x,self.D,Y);
            U = prox(Y-d*dJdY,d*self.lam);
            jU = J(x,self.D,X);
            tt = 0;
            
            while jU > jY + np.sum(dJdY*(U-Y)) + 0.5*np.sum(np.power(U-Y,2.0))/d:
                d *= 0.5;
                the_n = root(dt,the,d);
                
                Y = (1-the_n)*X+the_n*V;
                jY = J(x,self.D,Y);
                dJdY = dJdA(x,self.D,Y);
                U = prox(Y-d*dJdY,d*self.lam);
                jU = J(x,self.D,U);
                tt += 1;
                if tt > 10:
                    break;
            
            dt = d;
            the = the_n;
            X_n = U if F(x,self.D,U,self.lam) <= F(x,self.D,X,self.lam) else X;
            V   = X + (U-X)/the;
            X   = np.copy(X_n);
            
            print it, t, dt, the, J(x,self.D,X), F(x,self.D,X,self.lam);
            t+=1;
            
        f = F(x,self.D,X,self.lam);
        return X,f;
    
    def step_D(self,dt,iit):
        for i in range(self.X.shape[0]):
            x = self.X[i];
            A,f = self.step_A(x,(iit,i));
            print iit,i,f;
            self.dD += get_corr(get_conv(A,self.D)-x,A)/(self.IT);
            self.it += 1;
            if self.it % self.IT == 0:
                self.D -= dt*self.dD;                
                for l in range(self.L):
                    #self.D[l] -= np.mean(self.D[l]);
                    self.D[l] /= np.linalg.norm(self.D[l])+1e-1;
                self.dD.fill(0.0);