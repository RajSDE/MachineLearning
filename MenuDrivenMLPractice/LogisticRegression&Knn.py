# B. Write a menu-driven program in python to perform.
#   1. Logistic Regression.
#   2. K-Nearest Neighbour Classification.

# defining functions  
def read_dataset():
    df = pd.read_csv('User_Data2.csv')
    return df
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
    print('======================================================')
    print(df)
    print('======================================================')
    time.sleep(1)
   
def logistic_regression():
    print('\nApplying Logistic Regression', end="")
    time.sleep(1)
    print('.', end="")
    time.sleep(1)
    print(' .',end="")
    time.sleep(1)
    print(' .')
    read_dataset()
    time.sleep(1) 
    print('--------------------------------')
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
    time.sleep(2)
def knn():
    print('\nApplying K-Nearest Neighbour Classifier', end="")
    time.sleep(1)
    print('.', end="")
    time.sleep(1)
    print(' .',end="")
    time.sleep(1)
    print(' .')
    read_dataset()
    time.sleep(1) 
    print('--------------------------------')
    from sklearn.neighbors import KNeighborsClassifier

    classifier = KNeighborsClassifier(n_neighbors=8)
    classifier.fit(x_train, y_train)

    y_predit = classifier.predict(x_test)
    print(y_test)
    print(y_predit)
    from sklearn import metrics
    print('Prediction Accuracy = ', np.round(metrics.accuracy_score(y_test,y_predit)*100, 2),'%')
    time.sleep(2)

# 1. Write a program in Python simulate Logistic Regression on a CSV_DATASET.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

df = read_dataset()
# input
x = df[['Age','EstimatedSalary']].values
# output
y = df['Purchased'].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=0)
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)


print("Logistic Regression & KNN")  

# creating options  
while True:  
    print("\nMAIN MENU")
    print('-------------------')  
    print("1. Read the Dataset")  
    print("2. Display the Dataset")
    print("3. Apply Logistic Regression")
    print("4. Apply K-Nearest Neighbour Classifier ") 
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
        logistic_regression()
    elif choice1 == 4:
        knn()
    elif choice1 == 5:
        break      
    else:  
        print("Oops! Incorrect Choice.")  