# Write a Program in python to simulate KNN classification on CSV dataset.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Raj_1901227387.csv')
print('Dataset')
print('======================================================')
print(df)
print('======================================================')

# input
x = df[['Age','EstimatedSalary']].values
# output
y = df['Purchased'].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=0)

'''
print('Before Scaling') 
print('Xtrain') 
print(x_train) 
print("XTest") 
print(x_test)
'''


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)

# Use Classifier
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=8)
classifier.fit(x_train, y_train)

y_predit = classifier.predict(x_test)
print(y_test)
print(y_predit)
from sklearn import metrics
print('Prediction Accuracy = ', np.round(metrics.accuracy_score(y_test,y_predit)*100, 2),'%')