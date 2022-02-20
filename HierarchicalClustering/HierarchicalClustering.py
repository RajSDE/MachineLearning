# Simulation of Hierarchical Clustering on custom design dataset

import pandas as pd
from matplotlib import pyplot as plt
import scipy.cluster.hierarchy as sch
dataset = pd.read_csv('customData.csv')
X =dataset.iloc[:,[3,4]].values

print(X)
dendogram = sch.dendrogram(sch.linkage(X,method='single'))
plt.title('SingleLinkage')
plt.show()