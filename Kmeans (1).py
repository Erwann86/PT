# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 13:44:52 2022

@author: damii
"""

import numpy as np
from scipy.spatial.distance import cdist 
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
#from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import time

def crea_matrice(nb_données, taille_données, val_min, val_max):
    Data = np.random.randint(val_min, val_max, size =(nb_données, taille_données))
    return(Data)


Matrice = crea_matrice(1000,3,0,1000)


def kmeans(x,k, no_of_iterations):
    start = time.time()
    rep = np.random.choice(len(x), k, replace=False)
    #Randomly choosing Centroids 
    centroids = x[rep, :] #Step 1
    
    #finding the distance between centroids and all the data points
    #distances = cdist(x, centroids ,'euclidean') #Step 2
    
    distances =np.array([
        np.array([
            np.linalg.norm(x[ligne,:]-centroids[i,:]) for i in range(k)
        ]) 
        for ligne in range (x.shape[0])
    ])
     
    #Centroid with the minimum Distance
    points = np.array([np.argmin(i) for i in distances]) #Step 3
     
    #Repeating the above steps for a defined number of iterations
    #Step 4
    for _ in range(no_of_iterations): 
        centroids = []
        for rep in range(k):
            #Updating Centroids by taking mean of Cluster it belongs to
            temp_cent = x[points==rep].mean(axis=0) 
            centroids.append(temp_cent)
 
        centroids = np.vstack(centroids) #Updated Centroids 
        print("centroids")
        print(centroids) 
        distances = cdist(x, centroids ,'euclidean')
        points = np.array([np.argmin(i) for i in distances])
        
        distances_intercluster = cdist(centroids, centroids, 'euclidean')
        
        end = time.time()
         
    return {'label':points,'class':k,'temp': end - start, 'intercluster': distances_intercluster, 'distances': distances}


data = load_digits().data
pca = PCA(2)

df = pca.fit_transform(Matrice)


results = kmeans(df,3,10)

label = results['label']
print(results['intercluster'])
print(results['label'])
print(results['class'])
print(results['temp'])
print(results['distances'])


u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(df[label == i , 0] , df[label == i , 1] , label = i)
plt.legend()
plt.show()