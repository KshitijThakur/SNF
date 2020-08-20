import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io
import os

def rho_func(d1,p):
    d1sq = np.sum(d1**2,1).reshape(p,1)
    A = np.repeat(d1sq,p,1)
    B = A.transpose()
    C = np.dot(d1,d1.transpose())
    rhosq = A + B - 2*C
    rhosq[rhosq == 0] = rhosq.min()*1.0e-2
    rrho=np.sqrt(np.abs(rhosq))
    return rhosq

def Wmatrix(rhosq,p):
    K = 20
    rrho =np.sqrt(np.abs(rhosq))
    D = np.sort(rrho)
    E = np.sum(D[:,0:K+1],1).reshape(p,1)
    F = E/K
    xini = np.repeat(F,p,1)
    xjnj = xini.transpose()
    e = (xini + xjnj + rrho)/3
    u=0.5
    G = -(rhosq/(u*e))
    W = np.exp(G)
    return W

def Pmatrix(W,p):
    den = np.sum(W,1).reshape(p,1) - np.diag(W).reshape(p,1)
    den[den < 1.0e-16] = 1.0
    #X=np.repeat(0.5,p)
    #Y = np.diag(X)
    Q = ((W/2)/den)
    P = Q-np.diag(np.diag(Q))+0.5*np.eye(p)
    #np.diag(P) = 0.5
    return P

def Smatrix(W,p):
    S = np.zeros((p,p))
    k = 20
    Wsort = np.sort(W,1)
    Wknn = Wsort[:,0:k+1]
    Wid = np.argsort(W,1)
    Widknn = Wid[:,0:k+1]
    Sden= np.sum(Wknn,1).reshape(p,1)
    Sneu = Wknn
    Snonzero = Sneu/Sden
    for col in range (0,p):
        S[col, Widknn[col,:]]= Snonzero[col, :]
    return S

def load_data():    
    #data = scipy.io.loadmat("../Data/simulation.mat")
    #data1 = data["data1"]
    #data2 = data["data2"]
    data_1 = pd.read_csv("../Data/LGG/miRNA267" , delimiter = " ")
    data_2 = pd.read_csv("../Data/LGG/RNA267" , delimiter = " ")
    data_3 = pd.read_csv("../Data/LGG/mDNA267" , delimiter = " ")
    data_4 = pd.read_csv("../Data/LGG/RPPA267" , delimiter = " ")
    a1 = np.array(data_1)
    a2 = np.array(data_2)
    a3 = np.array(data_3)
    a4 = np.array(data_4)
    #check for transpose
    dat1 = a1[1:, 1:]
    dat2 = a2[1:, 1:]
    dat3 = a3[1:, 1:]
    dat4 = a4[1:, 1:]
    data1 = np.array(dat1 , dtype = np.float64)
    data2 = np.array(dat2 , dtype = np.float64)
    data3 = np.array(dat3 , dtype = np.float64)
    data4 = np.array(dat4 , dtype = np.float64)
    x = np.abs(data1.mean(0))
    y = np.abs(data2.mean(0))
    z = np.abs(data3.mean(0))
    w = np.abs(data4.mean(0))
    a = np.abs(data1.var(0))
    b = np.abs(data2.var(0))
    c = np.abs(data3.var(0))
    d = np.abs(data4.var(0))
    d1 = (data1 - x)/(np.sqrt(a))
    d2 = (data2 - y)/(np.sqrt(b))
    d3 = (data3 - z)/(np.sqrt(c))
    d4 = (data4 - w)/(np.sqrt(d))
    return [d1,d2]
            
def itr():
    t = 10
    data = load_data() 
    n_data = len(data)
    p,m = data[0].shape
    P = np.zeros((n_data,p,p))
    S = np.zeros((p,p))
    W = np.zeros((p,p))
    rhosq = np.zeros((p,p))
    #Loop to calculate Pmatrix for all n data
    for i in range(0,n_data):
       rhosq = rho_func(data[i],p) 
       W = Wmatrix(rhosq,p)
       P[i,:,:] = Pmatrix(W,p)

    #loop to calculate P
    sum_indx = np.arange(n_data)
    Pfinal = P.copy()
    for it in range(0,t):
        for v in range(0,n_data):
           rhosq = rho_func(data[v],p) 
           W = Wmatrix(rhosq,p)
           S = Smatrix(P[v,:,:],p)
           l = list(filter(lambda x : x!=v , sum_indx))
           Psum = P[l,:,:].sum(axis = 0)/(n_data-1)
           Pfinal[v,:,:] = np.dot(np.dot(S,Psum),S.transpose())
           Pfinal[v,:,:] = Pmatrix(Pfinal[v,:,:],p)
        Pans = Pfinal.mean(axis=0)
    return Pans

