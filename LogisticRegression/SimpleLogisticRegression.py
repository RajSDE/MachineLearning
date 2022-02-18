# Write a program in Python simulate Logistic Regression on a CSV_DATASET.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('User_Data.csv')
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

'''
print('After Scaling') 
print('Xtrain') 
print(x_train) 
print('x_test') 
print(x_test)
'''

from sklearn import linear_model
Lreg = linear_model.LogisticRegression(random_state=0)
Lreg.fit(x_train, y_train)

# Predict Based on testing model
y_pred = Lreg.predict(x_test)
print('Y-Test')
print(y_test.reshape(1,-1))
print('Y-Predicted')
print(y_pred.reshape(1,-1))

from sklearn import metrics
print('Accuracy of Logistic REgression model is: ', metrics.accuracy_score(y_test, y_pred)*100)
