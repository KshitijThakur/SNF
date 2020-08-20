from sklearn.cluster import SpectralClustering
import numpy as np
import pandas as pd

#df=pd.read_csv("../SNF.txt",header=None,sep=" ").to_numpy()
df=pd.read_csv("../SNF.txt",header=None,sep=" ")
x = df.shape[0]
s,u=np.linalg.eigh(np.identity(x)-df)
U=u[:,-2:].copy()

t=U[:,1].copy()
U[:,1]=U[:,0].copy()
U[:,0]=t.T.copy()

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
labels_pred = KMeans(n_clusters=3, random_state=0).fit(U[:,:]).predict(U[:,:])
s_score = silhouette_score(U[:,:], labels_pred)

#print(s_score)
#print(labels_pred)
