# Horizontal and Vertical stack and simulation of K-mean clustering on custom design dataset

import numpy as np
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans

X = np.random.randint(25,50,(25,2))
Y = np.random.randint(55,85,(25,2))

Z = np.vstack((X,Y))
print(Z)
plt.scatter(Z[:,0],Z[:,1],c ='red')

plt.show()
KMeans = KMeans(n_clusters=2)
KMeans.fit(Z)
y_means = KMeans.predict(Z)
center = KMeans.cluster_centers_
print(center)
plt.scatter(Z[:,0], Z[:,1],y_means)
plt.scatter(center[:,0],center[:,1], c ='red')
plt.show()