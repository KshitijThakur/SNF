import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io
#def load_data(fname1):
    #data1 = pd.read_csv(fname1, " ")
    
    #return np.array(data1)

#data1 = load_data("total_mi_cer")
#data2 = load_data("total_mr_cer")
#d1 = data1[0:100,:].transpose()
#d2 = data2[0:100,:].transpose()
#p = patient , m = measurement
def load_data():    
    data = scipy.io.loadmat("../Data/simulation.mat")
    d1 = data["data1"]
    d2 = data["data2"]
    return [d1,d2]
def smooth(d1):
    a = 0.5
    p,m = d1.shape
    d1sq = np.sum(d1**2,1).reshape(p,1)
    X = np.repeat(d1sq,p,1)
    Y = X.transpose()
    Z = np.dot(d1,d1.transpose())
    M = X + Y - 2*Z
    W = np.sqrt(np.abs(M))
    V = np.sum(W , axis = 1)
    D = np.diag(V**(-0.5))
    WW = np.dot(np.dot(D,W),D)
    #return WW, D , W
    d = d1
    Fo = np.ones((p,m))
    Fo[d<0.5] = 0
    Ft = d1 
    tol = 1
    while(tol>1.0e-6):
        Ftp1 = a*np.dot(WW,Ft)+(1-a)*Fo
        tol = np.linalg.norm(Ftp1 - Ft)
        print(tol)
        Ft = Ftp1
    return Ftp1
#data1 = smooth(d1)
#WW, D, W = smooth(d1)
