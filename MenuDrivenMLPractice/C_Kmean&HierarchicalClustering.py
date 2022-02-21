# A. Write a menu-driven program in python to perform.
#   1. Simple Linear regression.
#   2. Multiple Linear regression


# defining functions  
def read_dataset():
    dataset = pd.read_csv('Raj_1901227387.csv')
    return dataset
def display_dataset():
    print('\nFetching Dataset', end="")
    time.sleep(1)
    print('.', end="")
    time.sleep(1)
    print(' .',end="")
    time.sleep(1)
    print(' .')
    read_dataset()
    time.sleep(1)    
    print('Dataset')
    print('====================================================================')
    print(dataset)
    print('====================================================================')
    time.sleep(1)
def kmeanC():
    print('\nApplying K-Means Clustering', end="")
    time.sleep(1)
    print('.', end="")
    time.sleep(1)
    print(' .',end="")
    time.sleep(1)
    print(' .')
    read_dataset()
    time.sleep(1) 
    print('--------------------------------')
    from sklearn.cluster import KMeans

    X = dataset.iloc[:,[3,4]].values
    X = np.array(X).reshape(-1,2)

    plt.scatter(X[:,0],X[:,1],c ='red')
    plt.show()
    KMeans = KMeans(n_clusters=2)
    KMeans.fit(X)
    y_means = KMeans.predict(X)
    center = KMeans.cluster_centers_
    print(center)
    plt.scatter(X[:,0], X[:,1],y_means)
    plt.scatter(center[:,0],center[:,1], c ='red')
    plt.show()
    time.sleep(2)
def hClustering():
    print('\nApplying Hierarchical Clustering', end="")
    time.sleep(1)
    print('.', end="")
    time.sleep(1)
    print(' .',end="")
    time.sleep(1)
    print(' .')
    read_dataset()
    time.sleep(1) 
    print('--------------------------------')
    import scipy.cluster.hierarchy as sch
    X = dataset.iloc[:,[3,4]].values
    dendogram = sch.dendrogram(sch.linkage(X,method='single'))
    plt.title('SingleLinkage')
    plt.show()
    time.sleep(2)
def display_results():
    resultsKMean = kmeanC()
    print(resultsKMean)

# Importing required packages
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import time
dataset = read_dataset()


# Python Code for User Interface
print("K-means Clustering & Hierarchical Clustering")

# creating options  
while True:  
    print("\nMAIN MENU")
    print('-------------------')  
    print("1. Read the Dataset")  
    print("2. Display the Dataset")
    print("3. Apply K-Mean Clustering")
    print("4. Apply Hierarchical Clustering") 
    print("5. Exit")
    choice1 = int(input("\nEnter the Choice:"))  
  
    if choice1 == 1:  
        print('\nReading the Dataset', end="")
        time.sleep(1)
        print('.', end="")
        time.sleep(1)
        print(' .',end="")
        time.sleep(1)
        print(' .')
        read_dataset()
        time.sleep(1)
        print('Dataset has been loaded')
    elif choice1 == 2:
        read_dataset()  
        display_dataset()
    elif choice1 == 3:
        kmeanC()
    elif choice1 == 4:
        hClustering()
    elif choice1 == 5:
        break      
    else:  
        print("Oops! Incorrect Choice.")  